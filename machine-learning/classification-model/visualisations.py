import torch
import wandb
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging

# Configure logging
logger = logging.getLogger(__name__)

def visualize_feature_space(model, dataloader, device, class_names=None, max_samples=1000):
    """Visualizes feature embeddings with stratified sampling"""
    model.eval()
    embeddings = []
    labels = []
    
    # 1. Collect all embeddings and labels safely
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            try:
                # Use dedicated embedding method
                embeds = model.get_embeddings(inputs).cpu().numpy()
            except AttributeError:
                # Fallback for models without get_embeddings()
                embeds = model.features(inputs).flatten(1).cpu().numpy()
                
            embeddings.append(embeds)
            labels.append(targets.cpu().numpy())  # Ensure CPU conversion
    
    # Handle empty batches
    if not embeddings:
        try:
            wandb.log({"feature_space_error": "No data to visualize"})
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")
        return

    try:
        embeddings = np.concatenate(embeddings)
        labels = np.concatenate(labels)
    except ValueError:
        try:
            wandb.log({"feature_space_error": "Concatenation failed - check data dimensions"})
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")
        return

    # 2. Adaptive stratified sampling with safety checks
    try:
        if len(embeddings) > max_samples:
            from sklearn.model_selection import train_test_split
            _, embeds_subsample, _, labels_subsample = train_test_split(
                embeddings, labels,
                train_size=max_samples,
                stratify=labels,
                random_state=42
            )
        else:
            embeds_subsample, labels_subsample = embeddings, labels
    except Exception as e:
        try:
            wandb.log({"feature_space_error": f"Sampling failed: {str(e)}"})
        except Exception as we:
            logger.warning(f"Could not log to wandb: {we}")
        return

    # 3. TSNE with dynamic perplexity
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=max(5, min(30, len(embeds_subsample)//5)),  # Keep ≥5
            metric='cosine',
            max_iter=1000,
            random_state=42
        )
        projections = tsne.fit_transform(embeds_subsample)
    except Exception as e:
        try:
            wandb.log({"feature_space_error": f"TSNE failed: {str(e)}"})
        except Exception as we:
            logger.warning(f"Could not log to wandb: {we}")
        return

    # 4. Plotting with proper class names
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0], projections[:, 1],
                          c=labels_subsample, cmap='viridis', alpha=0.6,
                          edgecolors='w', linewidth=0.5)

    # Handle class names robustly
    if class_names and (len(class_names) >= (np.max(labels_subsample) + 1)):
        legend_labels = class_names
    else:
        legend_labels = [f'Class {i}' for i in range(np.max(labels_subsample) + 1)]

    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=legend_labels,
        title="Movement Classes",
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    # Add silhouette score
    if len(np.unique(labels_subsample)) > 1:
        score = silhouette_score(embeds_subsample, labels_subsample)
        plt.title(f"Feature Space (Silhouette: {score:.2f})")
    else:
        plt.title("Feature Space (Single Class)")

    # Proper W&B logging
    try:
        wandb.log({"feature_space": wandb.Image(plt)})
    except Exception as e:
        logger.warning(f"Could not log to wandb: {e}")
        # Save the plot locally as a fallback
        plt.savefig('feature_space.png', bbox_inches='tight')
        logger.info("Feature space visualization saved locally as 'feature_space.png'")
    
    plt.close()