import os
if os.name == 'nt':  # Windows
    os.environ["WANDB_SYMLINK"] = "false"
    os.environ["WANDB_ALWAYS_COPY"] = "true"
import torch
from torchvision.models import vgg16, resnet50, vit_b_16, VGG16_Weights, ResNet50_Weights, ViT_B_16_Weights
from torch import nn 
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader
import wandb
import platform
import torch.nn.functional as F 
from visualisations import visualize_feature_space

class ClassTokenSelector(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[:, 0]

class DirectionLoss(nn.Module):
    def __init__(self, class_weights, opposite_pairs, margin=0.5, pair_weight=0.3):
        super().__init__()
        self.class_weights = class_weights
        self.opposite_pairs = opposite_pairs
        self.margin = margin
        self.pair_weight = pair_weight
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        
    def forward(self, outputs, targets):
        ce_loss = self.ce_loss(outputs, targets)
        pair_loss = 0
        
        for i, j in self.opposite_pairs:
            mask = (targets == i) | (targets == j)
            if mask.sum() == 0:
                continue
            selected_outputs = outputs[mask]
            selected_targets = targets[mask]
            diff = selected_outputs[:, i] - selected_outputs[:, j]
            diff = torch.where(selected_targets == i, diff, -diff)
            pair_loss += F.relu(self.margin - diff).mean()
                    
        total_loss = ce_loss + self.pair_weight * pair_loss
        return total_loss

class ViTFeatures(nn.Module):
    def __init__(self, vit_model, dropout=0.1):
        super().__init__()
        self.class_token = vit_model.class_token
        original_conv = vit_model.conv_proj
        # Modified for 2-channel input
        self.conv_proj = nn.Conv2d(
            2,  # Changed to two channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )
        
        if vit_model.state_dict():  # Using pretrained weights
            with torch.no_grad():
                # Average RGB weights and expand for two channels
                radar_weights = original_conv.weight.mean(dim=1, keepdim=True)
                radar_weights = radar_weights.repeat(1, 2, 1, 1) / 2.0  # Adjusted for two channels
                std_ratio = original_conv.weight.std() / radar_weights.std()
                self.conv_proj.weight.copy_(radar_weights * std_ratio)
        else:
            nn.init.kaiming_normal_(self.conv_proj.weight, mode='fan_out', nonlinearity='relu')

        self.pos_dropout = nn.Dropout(p=dropout)
        self.encoder = vit_model.encoder
        self.pos_embedding = vit_model.encoder.pos_embedding

    def forward(self, x):
        # Input: [B, 2, 224, 224]
        x = self.conv_proj(x)  # [B, 768, 14, 14] 
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        return self.encoder(x)

class HeadRadarModel(nn.Module):
    def __init__(self, features, avgpool, classifier):
        super().__init__()
        self.features = features
        self.avgpool = avgpool
        self.classifier = classifier
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_embeddings(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = self.avgpool(x)
            return torch.flatten(x, 1)

def load_modified_model(model_type, num_classes, use_pretrained=True, hidden_dims=[256, 128], 
                        use_dropout=False, dropout_rate=0.3, use_batchnorm=False, num_unfrozen_layers=0):
    if model_type == 'vgg16':
        model = vgg16(weights=VGG16_Weights.DEFAULT if use_pretrained else None)
        original_weight = model.features[0].weight.data
        # Modified for 2-channel input
        new_first_layer = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        if use_pretrained:
            # Average and expand to two channels
            mean_weights = original_weight.mean(dim=1, keepdim=True)
            new_weights = mean_weights.repeat(1, 2, 1, 1) / 2.0
            new_first_layer.weight.data = new_weights
        else:
            nn.init.kaiming_normal_(new_first_layer.weight.data, mode='fan_out', nonlinearity='relu')
        model.features[0] = new_first_layer
        features = nn.Sequential(*list(model.features.children())[:-1])
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = 512
        
    elif model_type == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT if use_pretrained else None)
        original_weight = model.conv1.weight.data
        # Modified for 2-channel input
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_pretrained:
            # Average and expand to two channels
            mean_weights = original_weight.mean(dim=1, keepdim=True)
            new_weights = mean_weights.repeat(1, 2, 1, 1) / 2.0
            model.conv1.weight.data = new_weights
        else:
            nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        features = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4
        )
        avgpool = model.avgpool
        in_features = 2048

        if use_pretrained and num_unfrozen_layers > 0:
            layers = [model.layer4, model.layer3, model.layer2, model.layer1]
            for layer in layers[:num_unfrozen_layers]:
                for param in layer.parameters():
                    param.requires_grad = True

    elif model_type == 'vit':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if use_pretrained else None)
        original_conv = model.conv_proj
        # Modified for 2-channel input handled in ViTFeatures
        features = ViTFeatures(model)
        avgpool = ClassTokenSelector()
        in_features = 768
        
        if use_pretrained:
            for param in features.parameters():
                param.requires_grad = False
            if num_unfrozen_layers > 0:
                total_layers = len(features.encoder.layers)
                start_layer = total_layers - num_unfrozen_layers
                for i in range(start_layer, total_layers):
                    for param in features.encoder.layers[i].parameters():
                        param.requires_grad = True
    
    if use_pretrained and num_unfrozen_layers == 0 and model_type != 'vit':
        for param in features.parameters():
            param.requires_grad = False

    classifier_layers = []
    current_dim = in_features
    for dim in hidden_dims:
        classifier_layers.append(nn.Linear(current_dim, dim))
        classifier_layers.append(nn.ReLU())
        if use_batchnorm:
            classifier_layers.append(nn.BatchNorm1d(dim))
        if use_dropout:
            classifier_layers.append(nn.Dropout(p=dropout_rate))
        current_dim = dim
    classifier_layers.append(nn.Linear(current_dim, num_classes))
    classifier = nn.Sequential(*classifier_layers)

    return HeadRadarModel(features, avgpool, classifier)

# Remaining functions (get_class_names_from_subset, objective, bayesian_optimisation) remain unchanged

def get_class_names_from_subset(dataset):
    """Traverses through NormalizedDataset -> Subset -> original dataset"""
    if hasattr(dataset, 'subset'):
        if hasattr(dataset.subset, 'dataset') and hasattr(dataset.subset.dataset, 'classes'):
            return dataset.subset.dataset.classes
    return None

def objective(trial, train_dataset, val_dataset, device, class_weights, opposite_pairs, num_classes):
    """ 
    Objective function for optuna. Bayesian optimisation
    """
    """Objective function with W&B logging"""
    if platform.system() == "Windows":
        os.environ["WANDB_SYMLINK"] = "false"

    class_names = get_class_names_from_subset(val_dataset)

    # Initialize W&B run for this trial
    run = wandb.init(
        project="radar-head-movement",
        group="optuna-study",
        name=f"trial-{trial.number}",
        config={
            "trial_id": trial.number,
            "study_name": "head-movement-classification"
        }
    )

    try:

        # Get base parameters first
        model_type = trial.suggest_categorical('model_type', ['vgg16', 'resnet50', 'vit'])
        use_pretrained = trial.suggest_categorical('use_pretrained', [True, False])
        
        # Conditionally suggest num_unfrozen_layers
        num_unfrozen_layers = 0
        if use_pretrained:
            if model_type == 'resnet50':
                num_unfrozen_layers = trial.suggest_int('resnet_unfrozen_layers', 0, 4)
            elif model_type == 'vit':
                num_unfrozen_layers = trial.suggest_int('vit_unfrozen_layers', 0, 12)  # ViT-B has 12 layers

        params = {
            'model_type': model_type,
            'use_pretrained': use_pretrained,
            'num_unfrozen_layers': num_unfrozen_layers,
            'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'use_dropout': trial.suggest_categorical('use_dropout', [True, False]),
            'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
            'patience': trial.suggest_int('patience', 10, 30),
            'min_delta': trial.suggest_float('min_delta', 1e-5, 0.01, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'use_augmentation': trial.suggest_categorical('use_augmentation', [True, False]),
        }
        if params['use_augmentation']:
            params['noise_std'] = trial.suggest_float('noise_std', 0.0, 0.3)
        if params['optimizer'] == 'sgd':
            params['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        params['hidden_dims'] = [trial.suggest_categorical(f'layer_{i}_dim', [64, 128, 256, 512]) for i in range(params['num_layers'])]
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5) if params['use_dropout'] else 0.0

        wandb.config.update(params)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=(device.type == 'cuda'))
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], pin_memory=(device.type == 'cuda'))

        # Model
        model_params = {
            'model_type': params['model_type'],
            'use_pretrained': params['use_pretrained'],
            'hidden_dims': params['hidden_dims'],
            'use_dropout': params['use_dropout'],
            'dropout_rate': params['dropout_rate'],
            'use_batchnorm': params['use_batchnorm'],
            'num_unfrozen_layers': params.get('num_unfrozen_layers', 0),
        }
        model = load_modified_model(num_classes=5, **model_params).to(device)

        # Optimizer
        optimizer_config = {
            'params': filter(lambda p: p.requires_grad, model.parameters()),
            'lr': params['lr'],
            'weight_decay': params['weight_decay'],
        }
        if params['optimizer'] == 'sgd':
            optimizer_config['momentum'] = params['momentum']
            optimizer = optim.SGD(**optimizer_config)
        else:
            optimizer = optim.Adam(**optimizer_config)
        
        criterion = DirectionLoss(
            class_weights=class_weights,
            opposite_pairs=opposite_pairs,
            margin=0.5,
            pair_weight=0.3
        )

        # Training loop with augmentation
        best_val_acc = 0
        epochs_no_improve = 0
        best_weights = None
        
        for epoch in range(200):  # Increased max epochs
            model.train()
            train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if params['use_augmentation']:
                    inputs = inputs + torch.randn_like(inputs) * params['noise_std']
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            wandb.log({
            "train_loss": avg_train_loss,
            "epoch": epoch
            })

            # Validation
            model.eval()
            correct, total = 0, 0
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_loss += criterion(outputs, labels).item()
            val_acc = correct / total
            trial.report(val_acc, epoch)

            wandb.log({
                "val_acc": val_acc,
                "epoch": epoch,
                "val_loss": val_loss/len(val_loader) 
            })

            # Early stopping
            if val_acc > best_val_acc + params['min_delta']:
                best_val_acc = val_acc
                epochs_no_improve = 0
                best_weights = model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= params['patience']:
                    break
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if best_weights:
            model.load_state_dict(best_weights)

            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], pin_memory=(device.type == 'cuda'))
            visualize_feature_space(
                model=model,
                dataloader=val_loader,
                device=device,
                class_names=class_names
            )
            

        run.finish()
        return best_val_acc
    

    except Exception as e:
        run.finish()
        raise e

def bayesian_optimisation(train, val, device, class_weights, opposite_pairs, num_classes):
    # Optuna study with W&B callback
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
        multivariate=True,
        group=True,  # ← Better for conditional parameters
        n_startup_trials=50  
        ),
        pruner=None
        #pruner=optuna.pruners.HyperbandPruner(min_resource=30, 
        #                                      max_resource=50, 
        #                                      reduction_factor=2)
    )
    study.optimize(
        lambda trial: objective(trial, train, val, device, class_weights, opposite_pairs, num_classes),
        n_trials=150
    )

    wandb.init(project="radar-head-movement", name="final-model")

    # Best trial results
    print("\nBest trial:")
    trial = study.best_trial
    print(f"Validation accuracy: {trial.value:.4f}")
    print("Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Final model training
    best_params = trial.params

    return best_params