"""Utility functions: loss, seeding, and visualization."""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def seed_func(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContrastiveLoss(nn.Module):
    """Softmax cross-entropy loss for few-shot classification.
    
    Mathematically equivalent to: -log(exp(score_target) / sum(exp(scores)))
    """
    
    def forward(self, scores, targets):
        """
        Args:
            scores: (N, way_num) similarity scores
            targets: (N,) class labels
        """
        log_probs = torch.log_softmax(scores, dim=1)
        loss = -log_probs.gather(1, targets.view(-1, 1)).mean()
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
        device (str): device to use.
    """
    def __init__(self, num_classes=4, feat_dim=1600, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        
        # Normalize centers to unit sphere to match normalized features
        centers_norm = F.normalize(self.centers, p=2, dim=1)
        
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers_norm, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, centers_norm.t(), beta=1, alpha=-2)
        
        classes = torch.arange(self.num_classes).long().to(self.device)
            
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


def plot_confusion_matrix(targets, preds, num_classes=4, save_path=None, class_names=None):
    """
    Plot confusion matrix (IEEE format) - saves as PDF and PNG.
    
    Args:
        targets: Ground truth labels
        preds: Predicted labels
        num_classes: Number of classes
        save_path: Path to save the figure (without extension)
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['Corona', 'HF_NoPD', 'Surface', 'Void']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 14
    })
    
    cm = confusion_matrix(targets, preds)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    
    width = 7.16
    fig, ax = plt.subplots(figsize=(width, width))
    
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Greens',
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 14},
                vmin=0, square=True,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_xticklabels(class_names, fontsize=14, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=14, rotation=0)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
        plt.savefig(f"{base_path}.png", format='png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {base_path}.pdf and {base_path}.png')
    plt.close()


def plot_tsne(features, labels, num_classes=4, save_path=None):
    """
    t-SNE visualization of query features - saves as PDF and PNG.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 14
    })

    n = len(features)
    unique_n = len(np.unique(features, axis=0))
    print(f"t-SNE: Plotting {n} points (Unique: {unique_n})")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    n_components = min(30, n, features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print(f"  PCA reduced to {n_components} dimensions")
    
    perp = min(30, max(5, n // 3))
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca')
    embedded = tsne.fit_transform(features_pca)
    
    max_val = np.abs(embedded).max()
    if max_val > 0:
        embedded = embedded / max_val * 45
    
    width = 7.16
    plt.figure(figsize=(width, width))
    sns.set_style('white')
    
    scatter = sns.scatterplot(
        x=embedded[:, 0], y=embedded[:, 1],
        hue=labels, palette='bright',
        s=80, alpha=0.8, legend='full'
    )
    
    sns.despine()
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.title('t-SNE', fontsize=14, fontweight='bold')
    plt.xlabel('Dim 1', fontsize=14)
    plt.ylabel('Dim 2', fontsize=14)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        plt.savefig(f"{base_path}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
        plt.savefig(f"{base_path}.png", format='png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {base_path}.pdf and {base_path}.png')
    plt.close()


def plot_training_curves(history, save_path=None):
    """
    Plot combined train/val accuracy and loss curves.
    
    Args:
        history: dict with keys 'train_acc', 'val_acc', 'train_loss', 'val_loss'
        save_path: Path to save the figure (without extension)
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 11
    })
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    train_color = '#2E86AB'
    val_color = '#E94F37'
    
    # Accuracy Plot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_acc'], color=train_color, 
             linewidth=2, label='Train', marker='o', markersize=3)
    ax1.plot(epochs, history['val_acc'], color=val_color, 
             linewidth=2, linestyle='--', label='Validation', marker='s', markersize=3)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Loss Plot
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss'], color=train_color, 
             linewidth=2, label='Train', marker='o', markersize=3)
    ax2.plot(epochs, history['val_loss'], color=val_color, 
             linewidth=2, linestyle='--', label='Validation', marker='s', markersize=3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        full_path = f"{save_path}_curves.png" if not save_path.endswith('.png') else save_path
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {full_path}')
    
    plt.close()
    return fig