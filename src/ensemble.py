"""
Voting Ensemble for Multi-Model Classification

This module implements a soft voting ensemble that combines predictions
from multiple models (ViT + CNN) for improved accuracy.

Based on literature:
    [Abulfaraj & Binzagr, 2025] "A Deep Ensemble Learning Approach Based on a 
    Vision Transformer and Neural Network for Multi-Label Image Classification"
    BDCC 9(2):39, DOI: 10.3390/bdcc9020039

Key insight: Combining ViT + CNN in a voting ensemble achieves +2-4% improvement
over single models by leveraging complementary feature representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import List, Optional
import numpy as np


class VotingEnsemble:
    """
    Soft voting ensemble combining multiple models.
    
    Based on [Abulfaraj & Binzagr, 2025]: ensemble of ViT + CNN 
    outperforms single models by 2-4% on image classification tasks.
    
    Attributes:
        models: List of PyTorch models to ensemble
        weights: Optional weights for each model (default: equal weights)
        device: Device to run inference on
    
    Example:
        >>> ensemble = VotingEnsemble(
        ...     models=[vit_model, cnn_model],
        ...     weights=[1.2, 1.0],  # Favor ViT slightly
        ...     device='cuda'
        ... )
        >>> predictions = ensemble.predict(images)
    """
    
    def __init__(
        self, 
        models: List[nn.Module], 
        weights: Optional[List[float]] = None, 
        device: str = 'cuda'
    ):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of trained PyTorch models
            weights: Optional weights for each model (must sum to len(models))
            device: Device for inference ('cuda' or 'cpu')
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.device = device
        
        # Validate weights
        if len(self.weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(self.models)})")
        
        # Put all models in eval mode
        for model in self.models:
            model.eval()
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft voting: compute weighted average of model probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size, num_classes) with averaged probabilities
        """
        all_probs = []
        x = x.to(self.device)
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                output = model(x)
                probs = F.softmax(output, dim=1)
                all_probs.append(probs * weight)
        
        # Weighted average
        ensemble_prob = torch.stack(all_probs).sum(dim=0) / sum(self.weights)
        return ensemble_prob
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return predicted class indices.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor of shape (batch_size,) with predicted class indices
        """
        probs = self.predict_proba(x)
        return probs.argmax(dim=1)


def create_ensemble(
    vit_model: nn.Module,
    pancan_model: nn.Module,
    vgg_model: nn.Module,
    device: str = 'cuda',
    vit_weight: float = 1.2,
    pancan_weight: float = 1.0,
    vgg_weight: float = 1.0
) -> VotingEnsemble:
    """
    Create a 3-model voting ensemble following [Abulfaraj & Binzagr, 2025].
    
    Args:
        vit_model: Vision Transformer model
        pancan_model: PanCANLite model
        vgg_model: VGG16 baseline model
        device: Device for inference
        vit_weight: Weight for ViT (default 1.2, slightly favored)
        pancan_weight: Weight for PanCANLite
        vgg_weight: Weight for VGG16
        
    Returns:
        VotingEnsemble instance ready for inference
    """
    ensemble = VotingEnsemble(
        models=[vit_model, pancan_model, vgg_model],
        weights=[vit_weight, pancan_weight, vgg_weight],
        device=device
    )
    
    print("✅ Ensemble created with weights:")
    print(f"   - ViT-B/16:    {vit_weight}")
    print(f"   - PanCANLite:  {pancan_weight}")
    print(f"   - VGG16:       {vgg_weight}")
    print(f"\n📚 Based on: [Abulfaraj & Binzagr, 2025] BDCC 9(2):39")
    
    return ensemble


def evaluate_ensemble(
    ensemble: VotingEnsemble,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    class_names: List[str],
    individual_models: Optional[dict] = None
) -> dict:
    """
    Evaluate ensemble and optionally compare with individual models.
    
    Args:
        ensemble: VotingEnsemble instance
        test_loader: Test data loader
        device: Device for inference
        class_names: List of class names for reporting
        individual_models: Optional dict of {name: model} for comparison
        
    Returns:
        Dictionary with ensemble and individual model metrics
    """
    print("="*60)
    print("📊 ENSEMBLE EVALUATION ON TEST SET")
    print("="*60)
    
    ensemble_preds = []
    ensemble_labels = []
    
    # Collect individual predictions if models provided
    individual_preds = {name: [] for name in (individual_models or {})}
    
    for images, labels in test_loader:
        images = images.to(device)
        
        # Ensemble prediction
        preds = ensemble.predict(images)
        ensemble_preds.extend(preds.cpu().numpy())
        ensemble_labels.extend(labels.numpy())
        
        # Individual predictions
        if individual_models:
            for name, model in individual_models.items():
                with torch.no_grad():
                    individual_preds[name].extend(
                        model(images).argmax(dim=1).cpu().numpy()
                    )
    
    # Calculate metrics
    ensemble_acc = accuracy_score(ensemble_labels, ensemble_preds)
    ensemble_f1 = f1_score(ensemble_labels, ensemble_preds, average='weighted')
    
    results = {
        'ensemble': {
            'accuracy': ensemble_acc,
            'f1_score': ensemble_f1,
            'predictions': ensemble_preds,
            'labels': ensemble_labels
        }
    }
    
    # Calculate individual metrics
    if individual_models:
        print(f"\n🎯 RESULTS COMPARISON:")
        print(f"   {'Model':<20} {'Accuracy':<12} {'vs Ensemble':<12}")
        print(f"   {'-'*44}")
        
        for name in individual_models:
            acc = accuracy_score(ensemble_labels, individual_preds[name])
            f1 = f1_score(ensemble_labels, individual_preds[name], average='weighted')
            results[name] = {'accuracy': acc, 'f1_score': f1}
            diff = (ensemble_acc - acc) * 100
            print(f"   {name:<20} {acc:.2%}       {diff:+.2f}%")
        
        print(f"   {'-'*44}")
        print(f"   {'🏆 ENSEMBLE':<20} {ensemble_acc:.2%}")
    
    print(f"\n📈 Ensemble F1-Score: {ensemble_f1:.2%}")
    
    return results


def print_ensemble_comparison(
    results: dict,
    class_names: List[str]
) -> None:
    """
    Print detailed comparison between ensemble and individual models.
    
    Args:
        results: Results dictionary from evaluate_ensemble
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("📋 ENSEMBLE CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(
        results['ensemble']['labels'],
        results['ensemble']['predictions'],
        target_names=class_names
    )
    print(report)
