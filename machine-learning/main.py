import os
if os.name == 'nt':  # Windows
    os.environ["WANDB_SYMLINK"] = "false"
    os.environ["WANDB_ALWAYS_COPY"] = "true"
from sklearn.metrics import confusion_matrix
import numpy as np
from timefrequencydataset import TimeFrequencyMapDataset
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
import torch
import optuna
from model import objective, load_modified_model, DirectionLoss
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedShuffleSplit
import platform
from visualisations import visualize_feature_space


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if platform.system() == "Windows":
        os.environ["WANDB_SYMLINK"] = "false"

    # Initialize W&B project
    wandb.init(project="radar-head-movement", name="main-experiment")

   # Dataset and splits
    raw_dataset = TimeFrequencyMapDataset(os.path.join("clean_data"), compute_stats=False)
    targets = [raw_dataset[i][1].item() for i in range(len(raw_dataset))]
    print("Actual class order:", raw_dataset.classes)

    # --- Critical Fix: Proper Stratified Splitting ---
    # First split: Train+Val vs Test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(sss.split(np.zeros(len(raw_dataset)), targets))
    
    # Second split: Train vs Val (relative to train_val_indices)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rel_indices, val_rel_indices = next(sss.split(
        np.zeros(len(train_val_indices)), 
        [targets[i] for i in train_val_indices]
    ))
    
    # Map to original dataset indices
    train_abs_indices = train_val_indices[train_rel_indices]
    val_abs_indices = train_val_indices[val_rel_indices]

    # Create subsets
    train_subset = Subset(raw_dataset, train_abs_indices)
    val_subset = Subset(raw_dataset, val_abs_indices)
    test_subset = Subset(raw_dataset, test_indices)

    # --- Fix: Compute class weights BEFORE normalization ---
    labels = [train_subset[i][1].item() for i in range(len(train_subset))]  # Use subset directly
    class_counts = torch.bincount(torch.tensor(labels), minlength=5) + 1
    class_weights = (1.0 / class_counts.float())
    class_weights = (class_weights / class_weights.sum()).to(device)

    def calculate_stats(subset):
        tensors = torch.cat([subset[i][0] for i in range(len(subset))])
        mean = tensors.mean().item()
        std = tensors.std().item()
        return torch.tensor(mean), torch.tensor(std)

    train_mean, train_std = calculate_stats(train_subset)

    class NormalizedDataset(Dataset):
        def __init__(self, subset, mean, std):
            self.subset = subset
            self.mean = mean
            self.std = std

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return (x - self.mean) / self.std, y

    train = NormalizedDataset(train_subset, train_mean, train_std)
    val = NormalizedDataset(val_subset, train_mean, train_std)
    test = NormalizedDataset(test_subset, train_mean, train_std)


    class_to_idx = {name: i for i, name in enumerate(raw_dataset.classes)}
    opposite_pairs = [
        (class_to_idx["Left"], class_to_idx["Right"]),
        (class_to_idx["Down"], class_to_idx["Up"])
    ]
    print("\nClass Verification:")
    print(f"{'Class Name':<15} | {'Index':<5} | {'Opposite Pair'}")
    for name, idx in class_to_idx.items():
        opposites = [pair for pair in opposite_pairs if idx in pair]
        print(f"{name:<15} | {idx:<5} | {opposites}")

    criterion = DirectionLoss(
        class_weights=class_weights.to(device),
        opposite_pairs=opposite_pairs,
        margin=0.5,
        pair_weight=0.3
    ).to(device)

    # Optuna study with W&B callback
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
        multivariate=True,
        group=True  # ← Better for conditional parameters
        ),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=200, reduction_factor=3)
    )
    study.optimize(
        lambda trial: objective(trial, train, val, device, class_weights, opposite_pairs),
        n_trials=100
    )

    # Best trial results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"Validation accuracy: {trial.value:.4f}")
    print("Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Final model training
    best_params = trial.params
    model_params = {
    'model_type': best_params['model_type'],
    'use_pretrained': best_params['use_pretrained'],
    'hidden_dims': [best_params[f'layer_{i}_dim'] for i in range(best_params['num_layers'])],
    'use_dropout': best_params.get('use_dropout', False),
    'dropout_rate': best_params.get('dropout_rate', 0.0),
    'use_batchnorm': best_params.get('use_batchnorm', False),
    'num_unfrozen_layers': best_params.get('resnet_unfrozen_layers', best_params.get('vit_unfrozen_layers', 0))
    }
    
    model = load_modified_model(num_classes=5, **model_params).to(device)
    full_train = ConcatDataset([train, val])
    train_loader = DataLoader(full_train, 
                            batch_size=best_params['batch_size'], 
                            shuffle=True)
    test_loader = DataLoader(test, 
                           batch_size=best_params['batch_size'])
    val_loader = DataLoader(val, batch_size=best_params['batch_size'])

    optimizer_config = {
        'params': filter(lambda p: p.requires_grad, model.parameters()),
        'lr': best_params['lr'],
        'weight_decay': best_params['weight_decay']
    }
    if best_params['optimizer'] == 'sgd':
        optimizer_config['momentum'] = best_params['momentum']
        optimizer = optim.SGD(**optimizer_config)
    else:
        optimizer = optim.Adam(**optimizer_config)

    criterion = DirectionLoss(
        class_weights=class_weights.to(device),
        opposite_pairs=opposite_pairs,
        margin=0.5,
        pair_weight=0.3
    ).to(device)

    best_acc = 0
    patience = best_params['patience']
    no_improve = 0

    # Final training loop
    for epoch in tqdm(range(100), desc="Final Training"):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate loss
        wandb.log({"final_train_loss": epoch_loss/len(train_loader)})

        # Validation check
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct/len(val)

        wandb.log({
            "final_train_loss": epoch_loss/len(train_loader),
            "val_acc": val_acc
        })
                
        if val_acc > best_acc + 0.001:
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / total
    test_acc = correct / total
    print(f"\nFinal Test Accuracy: {100 * test_acc:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = raw_dataset.classes  # Get original class names
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Pretty-print confusion matrix
    print("\nConfusion Matrix (Normalized):")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("True \\ Predicted", *class_names, sep="\t")
    for i, (row, class_name) in enumerate(zip(cm_normalized, class_names)):
        print(f"{class_name:15}", *["%.2f" % val for val in row], sep="\t")

    # Final test logging
    wandb.log({
        "final_test_accuracy": test_acc,
        "final_test_loss": avg_test_loss,
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds,
            y_true=all_labels,
            class_names=class_names
        )
    })

    final_model_path = "final_head_movement_model.pth"
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)

    visualize_feature_space(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names 
    )

    # Then update the existing wandb.log() to:
    wandb.log({
        "final_test_accuracy": test_acc,
        "final_test_loss": avg_test_loss,
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=all_preds,
            y_true=all_labels,
            class_names=class_names
        ),
        "final_model": wandb.Table(
            columns=["Path"],
            data=[[final_model_path]]
        )
    })

if __name__ == "__main__":
    main()