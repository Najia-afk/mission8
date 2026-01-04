"""
VGG16 Baseline Model for E-commerce Image Classification

This module provides a VGG16-based classifier with frozen backbone,
following the same methodology as PanCANLite for fair comparison.

References:
    [Simonyan & Zisserman, 2015] Very Deep Convolutional Networks for Large-Scale Image Recognition
    [Jiu et al., 2025] PanCAN: Bridging Global Context with Local Details - arXiv:2512.23486
"""

import torch
import torch.nn as nn
import torchvision
from pathlib import Path


class VGG16Baseline(nn.Module):
    """
    VGG16 Baseline with frozen backbone (same approach as PanCAN).
    
    Architecture:
        - VGG16 features (frozen) - pretrained on ImageNet
        - Adaptive pooling → 7×7 feature map
        - Trainable classifier: 25088 → 4096 → 1024 → num_classes
    
    This serves as a strong baseline for comparison with context-aware
    architectures like PanCANLite [Jiu et al., 2025].
    """
    
    def __init__(self, num_classes: int, dropout: float = 0.5):
        """
        Initialize VGG16 baseline model.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # Load pretrained VGG16
        vgg = torchvision.models.vgg16(weights='IMAGENET1K_V1')
        
        # Freeze backbone
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Trainable classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
        
        # Calculate and store parameter counts
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.total_params = sum(p.numel() for p in self.parameters())
        
        print(f"VGG16 Baseline: {self.trainable_params:,} trainable / {self.total_params:,} total params")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with frozen backbone."""
        with torch.no_grad():
            x = self.features(x)
        x = self.classifier(x)
        return x


def load_or_create_vgg16(
    num_classes: int,
    models_dir: Path,
    device: torch.device,
    dropout: float = 0.5
) -> tuple:
    """
    Load existing VGG16 model or create new one.
    
    Args:
        num_classes: Number of output classes
        models_dir: Directory containing saved models
        device: Device to load model to
        dropout: Dropout probability
        
    Returns:
        Tuple of (model, skip_training_flag)
    """
    model = VGG16Baseline(num_classes, dropout=dropout)
    model_path = Path(models_dir) / 'vgg16_best.pt'
    
    if model_path.exists():
        print(f"✅ Found existing VGG16 model at {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return model, True
    else:
        print("📦 Will train VGG16 baseline.")
        model = model.to(device)
        return model, False


def evaluate_vgg16(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list = None
) -> dict:
    """
    Evaluate VGG16 model on test set.
    
    Args:
        model: Trained VGG16 model
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: Optional list of class names
        
    Returns:
        Dictionary with accuracy, f1_score, predictions, labels
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print("\n" + "="*60)
    print("VGG16 Baseline Test Results")
    print("="*60)
    print(f"Accuracy: {100*acc:.2f}%")
    print(f"F1 Score (macro): {100*f1:.2f}%")
    print("="*60)
    
    return {
        'accuracy': acc,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels
    }
