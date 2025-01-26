import os
if os.name == 'nt':  # Windows
    os.environ["WANDB_SYMLINK"] = "false"
    os.environ["WANDB_ALWAYS_COPY"] = "true"
from sklearn.metrics import confusion_matrix
import numpy as np
from timefrequencydataset import TimeFrequencyMapDataset
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import torch
import optuna
from model import objective, load_modified_model, DirectionLoss
from torch import nn 
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedShuffleSplit
import platform


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if platform.system() == "Windows":
        os.environ["WANDB_SYMLINK"] = "false"

    # Initialize W&B project
    wandb.init(project="radar-head-movement", name="main-experiment")

    # Dataset and splits
    dataset = TimeFrequencyMapDataset(os.path.join("clean_data"))
    print("Actual class order:", dataset.classes)
    targets = [dataset[i][1].item() for i in range(len(dataset))]

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(sss.split(np.zeros(len(dataset)), targets))
    train_val = torch.utils.data.Subset(dataset, train_val_idx)
    test = torch.utils.data.Subset(dataset, test_idx)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(train_val)), [targets[i] for i in train_val_idx]))
    train = torch.utils.data.Subset(train_val, train_idx)
    val = torch.utils.data.Subset(train_val, val_idx)

    # Compute class weights with Laplace smoothing
    labels = [train[i][1].item() for i in range(len(train))]
    class_counts = torch.bincount(torch.tensor(labels), minlength=5) + 1  # Add 1 for smoothing
    class_weights = (1.0 / class_counts.float()).to(device)
    class_weights = (class_weights / class_weights.sum()).to(device)

    opposite_pairs = [(1, 3), (0, 4)]

    criterion = DirectionLoss(
    class_weights=class_weights.to(device),
    opposite_pairs=opposite_pairs,
    margin=0.5,    # Minimum logit difference between opposites
    pair_weight=0.3 # Weight for the pair penalty term
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
        'hidden_dims': [best_params[f'layer_{i}_dim'] 
                       for i in range(best_params['num_layers'])],
        'use_dropout': best_params.get('use_dropout', False),
        'dropout_rate': best_params.get('dropout_rate', 0.0),
        'use_batchnorm': best_params.get('use_batchnorm', False)
    }
    
    model = load_modified_model(num_classes=5, **model_params).to(device)
    full_train = ConcatDataset([train, val])
    train_loader = DataLoader(full_train, 
                            batch_size=best_params['batch_size'], 
                            shuffle=True)
    test_loader = DataLoader(test, 
                           batch_size=best_params['batch_size'])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                         lr=best_params['lr'])
    criterion = DirectionLoss()

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
            optimizer.step()
            epoch_loss += loss.item()  # Accumulate loss
        wandb.log({"final_train_loss": epoch_loss/len(train_loader)})

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
    class_names = dataset.classes  # Get original class names
    
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