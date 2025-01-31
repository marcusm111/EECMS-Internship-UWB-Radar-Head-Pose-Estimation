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
from model import load_modified_model, DirectionLoss, bayesian_optimisation
from utils import create_normalised_subsets, load_config
import torch.optim as optim
import wandb
import platform
from visualisations import visualize_feature_space
torch.manual_seed(42)
np.random.seed(42)

def main():
    # Initialise 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if platform.system() == "Windows":
        os.environ["WANDB_SYMLINK"] = "false"
    config = load_config("config.yaml")

   # Load raw timefrequencymap dataset
    raw_dataset = TimeFrequencyMapDataset(config["clean_data"], compute_stats=False)

    # Split for training
    train, val, test, class_weights, opposite_pairs = create_normalised_subsets(raw_dataset, device)

    # Custom loss function to help differentiate between opposite pairs, left and right, up and down
    criterion = DirectionLoss(
        class_weights=class_weights.to(device),
        opposite_pairs=opposite_pairs,
        margin=0.5,
        pair_weight=0.3
    ).to(device)

    # Get model parameters
    if config["bayesian_optimisation"]:
        # Find and use best params
        print("Using bayesian optimisation")
        params = bayesian_optimisation(train, val, device, class_weights, opposite_pairs)
        model_params = {
        'model_type': params['model_type'],
        'use_pretrained': params['use_pretrained'],
        'hidden_dims': [params[f'layer_{i}_dim'] for i in range(params['num_layers'])],
        'use_dropout': params.get('use_dropout', False),
        'dropout_rate': params.get('dropout_rate', 0.0),
        'use_batchnorm': params.get('use_batchnorm', False),
        'num_unfrozen_layers': params.get('resnet_unfrozen_layers', params.get('vit_unfrozen_layers', 0))
    }
    else:
        # Use default params
        print("Using default values")
        params = config["default_params"]
        model_params = params["model_params"]

    # Load model type
    model = load_modified_model(num_classes=5, **model_params).to(device)
    full_train = ConcatDataset([train, val])
    train_loader = DataLoader(full_train, 
                            batch_size=params['batch_size'], 
                            shuffle=True,
                            pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test, 
                           batch_size=params['batch_size'],
                           pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val, batch_size=params['batch_size'],
                            pin_memory=(device.type == 'cuda'))

    optimizer_config = {
        'params': filter(lambda p: p.requires_grad, model.parameters()),
        'lr': params['lr'],
        'weight_decay': params['weight_decay']
    }
    if params['optimizer'] == 'sgd':
        optimizer_config['momentum'] = params['momentum']
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
    patience = params['patience']
    no_improve = 0
    
    wandb.init(project="radar-head-movement", name="final-model")

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