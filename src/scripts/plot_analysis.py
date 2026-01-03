"""
Model Analysis and Evaluation Visualizations for Mission 8
Includes confusion matrix, parameter analysis, and other evaluation plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_confusion_matrix(cm, class_names, save_dir=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_dir: Optional directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('PanCAN Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def analyze_parameter_distribution(model):
    """
    Analyze and print parameter distribution by module
    
    Args:
        model: PyTorch model to analyze
    """
    param_stats = {}
    
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = total - trainable
        
        param_stats[name] = {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }

    print("\n" + "="*70)
    print("Parameter Distribution by Module")
    print("="*70)
    print(f"{'Module':<25} {'Total':>12} {'Trainable':>12} {'Frozen':>12}")
    print("-"*70)

    total_all = 0
    trainable_all = 0

    for name, stats in param_stats.items():
        print(f"{name:<25} {stats['total']:>12,} {stats['trainable']:>12,} {stats['frozen']:>12,}")
        total_all += stats['total']
        trainable_all += stats['trainable']

    print("-"*70)
    print(f"{'TOTAL':<25} {total_all:>12,} {trainable_all:>12,} {total_all-trainable_all:>12,}")
    print("="*70)
    
    return param_stats


def analyze_param_sample_ratio(trainable_params, train_samples):
    """
    Analyze and warn about parameter to sample ratio
    
    Args:
        trainable_params: Number of trainable parameters
        train_samples: Number of training samples
    """
    ratio = trainable_params / train_samples

    print("\n" + "="*60)
    print("CRITICAL: Parameter/Sample Ratio Analysis")
    print("="*60)
    print(f"Training samples: {train_samples}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Parameter/Sample ratio: {ratio:,.0f}:1")
    print("-"*60)

    if ratio < 2000:
        print("✅ GOOD: Ratio < 2000:1 - Model complexity appropriate for dataset")
        status = "optimal"
    elif ratio < 10000:
        print("⚠️ WARNING: Ratio in 2000-10000:1 range - Monitor for overfitting")
        status = "acceptable"
    else:
        print("❌ CRITICAL: Ratio > 10000:1 - High overfitting risk!")
        print("   Consider using PanCANLite or reducing model complexity")
        status = "critical"

    print("="*60)
    
    return ratio, status


def verify_backbone_freeze(model):
    """
    Verify that the backbone is frozen
    
    Args:
        model: PanCAN model to verify
    """
    try:
        backbone_trainable = sum(
            p.numel() for p in model.feature_extractor.backbone.parameters() 
            if p.requires_grad
        )
    except AttributeError:
        print("⚠️ Cannot verify backbone - model structure different")
        return None

    print("\n" + "="*60)
    print("Backbone Freeze Verification")
    print("="*60)

    if backbone_trainable == 0:
        print("✅ VERIFIED: Backbone is completely frozen (0 trainable params)")
        print("   This matches the paper's methodology!")
        status = "frozen"
    else:
        print(f"❌ ERROR: Backbone has {backbone_trainable:,} trainable parameters!")
        print("   This violates the paper's architecture!")
        status = "trainable"

    print("="*60)
    
    return status


def print_classification_report(all_labels, all_preds, class_names):
    """
    Print detailed classification report
    
    Args:
        all_labels: True labels
        all_preds: Predicted labels
        class_names: List of class names
    """
    from sklearn.metrics import classification_report
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def print_test_metrics(test_acc, test_f1, test_precision, test_recall):
    """
    Print formatted test metrics
    
    Args:
        test_acc: Test accuracy
        test_f1: Test F1 score
        test_precision: Test precision
        test_recall: Test recall
    """
    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    print(f"Accuracy: {100*test_acc:.2f}%")
    print(f"F1 Score (macro): {100*test_f1:.2f}%")
    print(f"Precision (macro): {100*test_precision:.2f}%")
    print(f"Recall (macro): {100*test_recall:.2f}%")
    print("="*60)


if __name__ == "__main__":
    print("Model analysis and evaluation utilities for Mission 8")
    print("Functions available:")
    print("  - plot_confusion_matrix()")
    print("  - analyze_parameter_distribution()")
    print("  - analyze_param_sample_ratio()")
    print("  - verify_backbone_freeze()")
    print("  - print_classification_report()")
    print("  - print_test_metrics()")
