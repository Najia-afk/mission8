"""
Vision Transformer (ViT) Baseline for Mission 8
Provides ViT-based image classification as a Transformer comparison to CNN architectures.
"""

import torch
import torch.nn as nn
import torchvision
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


class ViTBaseline(nn.Module):
    """
    Vision Transformer Baseline using pretrained ViT-B/16.
    Frozen backbone with trainable classification head.
    """
    
    def __init__(self, num_classes: int, dropout: float = 0.5, freeze_backbone: bool = True):
        """
        Initialize ViT baseline model.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate for classifier
            freeze_backbone: Whether to freeze ViT backbone
        """
        super().__init__()
        
        # Load pretrained ViT-B/16 (Vision Transformer Base with 16x16 patches)
        self.vit = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Get the hidden dimension from ViT
        hidden_dim = self.vit.heads.head.in_features  # 768 for ViT-B
        
        # Replace classification head with custom trainable head
        self.vit.heads = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Print parameter counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        
        print(f"\n{'='*60}")
        print("Vision Transformer (ViT-B/16) Baseline")
        print(f"{'='*60}")
        print(f"Total parameters:     {total:>12,}")
        print(f"Frozen (backbone):    {frozen:>12,}")
        print(f"Trainable (head):     {trainable:>12,}")
        print(f"{'='*60}")
        
        self.trainable_params = trainable
        self.total_params = total
        
    def forward(self, x):
        """Forward pass through ViT."""
        return self.vit(x)


def create_vit_model(num_classes: int, dropout: float = 0.5, freeze_backbone: bool = True) -> ViTBaseline:
    """
    Create ViT baseline model.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout rate
        freeze_backbone: Whether to freeze backbone
        
    Returns:
        ViTBaseline model instance
    """
    return ViTBaseline(num_classes, dropout, freeze_backbone)


def load_or_create_vit(
    num_classes: int,
    models_dir: Path,
    device: torch.device,
    dropout: float = 0.5
) -> tuple:
    """
    Load existing ViT model or create new one.
    
    Args:
        num_classes: Number of output classes
        models_dir: Directory containing models
        device: Device to load model to
        dropout: Dropout rate
        
    Returns:
        Tuple of (model, skip_training: bool)
    """
    vit_model = create_vit_model(num_classes, dropout)
    vit_model_path = models_dir / 'vit_best.pt'
    
    if vit_model_path.exists():
        print(f"\n✅ Found existing ViT model at {vit_model_path}")
        checkpoint = torch.load(vit_model_path, map_location=device, weights_only=False)
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        vit_model = vit_model.to(device)
        return vit_model, True
    else:
        print("\n⏳ Will train ViT baseline from scratch.")
        return vit_model, False


def evaluate_vit(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list = None
) -> dict:
    """
    Evaluate ViT model on test set.
    
    Args:
        model: ViT model
        test_loader: Test data loader
        device: Device
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Print results
    print("\n" + "="*60)
    print("Vision Transformer (ViT-B/16) Results")
    print("="*60)
    print(f"Accuracy: {100*accuracy:.2f}%")
    print(f"F1 Score (macro): {100*f1:.2f}%")
    print("="*60)
    
    # Classification report
    if class_names:
        print("\nPer-class Performance:")
        print("-"*60)
        report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
        print(report)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def plot_vit_comparison(
    pancan_acc: float,
    pancan_f1: float,
    pancan_params: int,
    vgg_acc: float,
    vgg_f1: float,
    vgg_params: int,
    vit_acc: float,
    vit_f1: float,
    vit_params: int,
    save_dir: Path = None
) -> None:
    """
    Plot comparison between PanCANLite, VGG16, and ViT.
    
    Args:
        pancan_acc/f1/params: PanCANLite metrics
        vgg_acc/f1/params: VGG16 metrics
        vit_acc/f1/params: ViT metrics
        save_dir: Directory to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['PanCANLite\n(CNN)', 'VGG16\n(CNN)', 'ViT-B/16\n(Transformer)']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    # Plot 1: Accuracy
    accuracies = [pancan_acc * 100, vgg_acc * 100, vit_acc * 100]
    bars1 = axes[0].bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(accuracies) - 5, max(accuracies) + 5])
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].axhline(y=max(accuracies), color='gold', linestyle='--', alpha=0.7, label='Best')
    
    # Plot 2: F1 Score
    f1_scores = [pancan_f1 * 100, vgg_f1 * 100, vit_f1 * 100]
    bars2 = axes[1].bar(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([min(f1_scores) - 5, max(f1_scores) + 5])
    for bar, f1 in zip(bars2, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{f1:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 3: Parameters (log scale)
    params = [pancan_params, vgg_params, vit_params]
    bars3 = axes[2].bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Parameters', fontsize=12, fontweight='bold')
    axes[2].set_title('Model Size (Trainable Params)', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    for bar, p in zip(bars3, params):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                     f'{p/1e6:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'model_comparison_with_vit.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Saved comparison plot to {save_path}")
    
    plt.show()


def plot_vit_comparison_plotly(
    pancan_acc: float,
    pancan_f1: float,
    pancan_params: int,
    vgg_acc: float,
    vgg_f1: float,
    vgg_params: int,
    vit_acc: float,
    vit_f1: float,
    vit_params: int
):
    """
    Create interactive Plotly comparison between all three models.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    models = ['PanCANLite (CNN)', 'VGG16 (CNN)', 'ViT-B/16 (Transformer)']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Test Accuracy', 'F1 Score (Macro)', 'Trainable Parameters'],
        specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Accuracy
    accuracies = [pancan_acc * 100, vgg_acc * 100, vit_acc * 100]
    fig.add_trace(
        go.Bar(
            x=models, y=accuracies,
            marker_color=colors,
            text=[f'{a:.2f}%' for a in accuracies],
            textposition='outside',
            hovertemplate='%{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # F1 Score
    f1_scores = [pancan_f1 * 100, vgg_f1 * 100, vit_f1 * 100]
    fig.add_trace(
        go.Bar(
            x=models, y=f1_scores,
            marker_color=colors,
            text=[f'{f:.2f}%' for f in f1_scores],
            textposition='outside',
            hovertemplate='%{x}<br>F1 Score: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Parameters
    params = [pancan_params, vgg_params, vit_params]
    fig.add_trace(
        go.Bar(
            x=models, y=params,
            marker_color=colors,
            text=[f'{p/1e6:.1f}M' for p in params],
            textposition='outside',
            hovertemplate='%{x}<br>Parameters: %{y:,.0f}<extra></extra>'
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title=dict(
            text='<b>CNN vs Transformer Architecture Comparison</b><br>' +
                 '<sub>PanCANLite (Custom CNN) vs VGG16 (Deep CNN) vs ViT-B/16 (Transformer)</sub>',
            x=0.5,
            font=dict(size=18)
        ),
        showlegend=False,
        height=500,
        margin=dict(t=100)
    )
    
    # Update y-axis for parameters to log scale
    fig.update_yaxes(type='log', row=1, col=3)
    
    fig.show()
    
    return fig


def print_architecture_comparison():
    """Print a summary of CNN vs Transformer architecture differences."""
    print("\n" + "="*70)
    print("CNN vs TRANSFORMER ARCHITECTURE COMPARISON")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    CONVOLUTIONAL NEURAL NETWORKS (CNN)              │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Local receptive fields (kernel convolutions)                     │
│ ✓ Translation equivariance (built-in inductive bias)               │
│ ✓ Hierarchical feature extraction (low → high level)               │
│ ✓ Parameter efficient for images (weight sharing)                  │
│ ✓ Works well with limited data                                     │
│                                                                     │
│ Examples: ResNet, VGG, EfficientNet                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    VISION TRANSFORMERS (ViT)                        │
├─────────────────────────────────────────────────────────────────────┤
│ ✓ Global attention (all patches attend to all patches)             │
│ ✓ No inductive bias (learns spatial relations from data)           │
│ ✓ Self-attention captures long-range dependencies                  │
│ ✓ Highly scalable with data and compute                            │
│ ✓ State-of-the-art on large-scale datasets                         │
│                                                                     │
│ ⚠ Requires large datasets (millions of images)                     │
│ ⚠ Higher computational cost                                        │
│                                                                     │
│ Examples: ViT, DeiT, Swin Transformer, BEiT                        │
└─────────────────────────────────────────────────────────────────────┘
""")
    print("="*70)


def print_final_comparison(
    pancan_acc: float, pancan_f1: float, pancan_params: int,
    vgg_acc: float, vgg_f1: float, vgg_params: int,
    vit_acc: float, vit_f1: float, vit_params: int,
    train_samples: int = 629
):
    """Print final comparison table with all three models."""
    
    # Calculate ratios
    pancan_ratio = pancan_params / train_samples
    vgg_ratio = vgg_params / train_samples
    vit_ratio = vit_params / train_samples
    
    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON: CNN vs TRANSFORMER")
    print("="*80)
    print(f"{'Model':<22} {'Type':<12} {'Params':<12} {'Ratio':<10} {'Accuracy':<12} {'F1 Score'}")
    print("-"*80)
    print(f"{'PanCANLite':<22} {'CNN':<12} {pancan_params:>9,}   {pancan_ratio:>6.0f}:1   {100*pancan_acc:>6.2f}%      {100*pancan_f1:>6.2f}%")
    print(f"{'VGG16 Baseline':<22} {'CNN':<12} {vgg_params:>9,}   {vgg_ratio:>6.0f}:1   {100*vgg_acc:>6.2f}%      {100*vgg_f1:>6.2f}%")
    print(f"{'ViT-B/16':<22} {'Transformer':<12} {vit_params:>9,}   {vit_ratio:>6.0f}:1   {100*vit_acc:>6.2f}%      {100*vit_f1:>6.2f}%")
    print("="*80)
    
    # Find best model
    accuracies = {'PanCANLite': pancan_acc, 'VGG16': vgg_acc, 'ViT-B/16': vit_acc}
    best_model = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_model]
    
    print(f"\n🏆 Best Accuracy: {best_model} ({100*best_acc:.2f}%)")
    
    # Analysis
    if pancan_acc >= vgg_acc and pancan_acc >= vit_acc:
        print(f"\n✅ PanCANLite achieves best performance with smallest model!")
        print(f"   → Demonstrates that lightweight CNNs excel on small datasets")
        print(f"   → {pancan_params/vit_params:.1%} the size of ViT, same/better accuracy")
    elif vit_acc > pancan_acc and vit_acc > vgg_acc:
        print(f"\n🔮 ViT-B/16 achieves best performance!")
        print(f"   → Transformer benefits from ImageNet pretraining")
        print(f"   → Global attention captures product patterns effectively")
    else:
        print(f"\n📊 VGG16 achieves best performance")
        print(f"   → Deep CNN architecture effective for this task")
    
    print("\n" + "="*80)
