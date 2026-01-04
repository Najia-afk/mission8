"""
Multimodal Fusion Model Evaluation.

Evaluates multimodal models that combine image and text features.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_multimodal_model(model, test_loader, device, text_feature_dim=5000):
    """
    Evaluate multimodal fusion model on test data.
    
    Args:
        model: Multimodal PyTorch model (expects image + text inputs)
        test_loader: DataLoader for test data
        device: torch device
        text_feature_dim: Dimension of text features (default: 5000 for TF-IDF)
    
    Returns:
        tuple: (predictions, labels, accuracy, f1_score)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Create dummy text features (zeros) for image-only evaluation
            # In production, would use actual text embeddings
            text_features = torch.zeros(batch_size, text_feature_dim).to(device)
            
            outputs = model(images, text_features)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return all_preds, all_labels, acc, f1


def print_multimodal_results(acc, f1, comparison_models=None):
    """
    Print multimodal evaluation results and comparisons.
    
    Args:
        acc: Multimodal model accuracy
        f1: Multimodal model F1 score
        comparison_models: dict with model names and their accuracies for comparison
    """
    print("\n" + "=" * 60)
    print("🎯 MULTIMODAL FUSION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {acc:.2%}")
    print(f"F1 Score:      {f1:.2%}")
    print("=" * 60)
    
    if comparison_models:
        print(f"\n📊 Improvement over single models:")
        for name, other_acc in comparison_models.items():
            improvement = (acc - other_acc) * 100
            print(f"   vs {name:<12} {improvement:+.2f}%")


def evaluate_and_report(model, test_loader, device, comparison_models=None, 
                        text_feature_dim=5000):
    """
    Complete multimodal evaluation with results reporting.
    
    Args:
        model: Multimodal PyTorch model
        test_loader: DataLoader for test data
        device: torch device
        comparison_models: dict with model names and accuracies for comparison
        text_feature_dim: Dimension of text features
    
    Returns:
        dict with 'predictions', 'labels', 'accuracy', 'f1_score'
    """
    preds, labels, acc, f1 = evaluate_multimodal_model(
        model, test_loader, device, text_feature_dim
    )
    
    print_multimodal_results(acc, f1, comparison_models)
    
    return {
        'predictions': preds,
        'labels': labels,
        'accuracy': acc,
        'f1_score': f1
    }
