"""
Metrics Evaluator for Model Performance.

Provides comprehensive metrics computation and evaluation
for classification models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    log_loss
)
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class MetricsEvaluator:
    """
    Comprehensive metrics evaluator for classification models.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            
            # F1 scores
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'micro_f1': f1_score(y_true, y_pred, average='micro'),
            
            # Precision and recall
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro'),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted'),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted'),
            
            # Advanced metrics
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            
            # Per-class metrics
            'per_class_f1': f1_score(y_true, y_pred, average=None).tolist(),
            'per_class_precision': precision_score(y_true, y_pred, average=None).tolist(),
            'per_class_recall': recall_score(y_true, y_pred, average=None).tolist(),
        }
        
        # Add log loss if probabilities are available
        if y_prob is not None:
            metrics['log_loss'] = log_loss(y_true, y_prob)
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: If True, return as dictionary
            
        Returns:
            Classification report (string or dict)
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4,
            output_dict=output_dict
        )
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', None)
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(
            y_true,
            y_pred,
            normalize=normalize
        )
    
    def compute_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Perform error analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Error analysis dictionary
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Find most confused class pairs
        confused_pairs = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.class_names[i],
                        'pred_class': self.class_names[j],
                        'count': int(cm[i, j]),
                        'rate': float(cm[i, j] / cm[i].sum())
                    })
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Find hardest and easiest classes
        class_difficulty = {
            self.class_names[i]: {
                'accuracy': float(per_class_accuracy[i]),
                'support': int(cm[i].sum()),
                'false_negatives': int(cm[i].sum() - cm[i, i]),
                'false_positives': int(cm[:, i].sum() - cm[i, i])
            }
            for i in range(self.num_classes)
        }
        
        return {
            'most_confused_pairs': confused_pairs[:10],
            'per_class_analysis': class_difficulty,
            'easiest_class': max(class_difficulty.items(), key=lambda x: x[1]['accuracy'])[0],
            'hardest_class': min(class_difficulty.items(), key=lambda x: x[1]['accuracy'])[0]
        }
    
    def format_metrics_table(self, metrics: Dict) -> str:
        """
        Format metrics as a readable table.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 50)
        lines.append("CLASSIFICATION METRICS")
        lines.append("=" * 50)
        
        # Global metrics
        lines.append("\n--- Global Metrics ---")
        global_metrics = [
            'accuracy', 'balanced_accuracy', 
            'macro_f1', 'weighted_f1',
            'macro_precision', 'macro_recall',
            'cohen_kappa', 'matthews_corrcoef'
        ]
        
        for metric in global_metrics:
            if metric in metrics:
                value = metrics[metric]
                lines.append(f"{metric:25s}: {value:.4f}")
        
        # Per-class metrics
        lines.append("\n--- Per-Class F1 Scores ---")
        if 'per_class_f1' in metrics:
            for i, f1 in enumerate(metrics['per_class_f1']):
                lines.append(f"{self.class_names[i]:25s}: {f1:.4f}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def save_metrics(
        self,
        metrics: Dict,
        output_path: Path,
        format: str = 'json'
    ):
        """
        Save metrics to file.
        
        Args:
            metrics: Dictionary of metrics
            output_path: Output file path
            format: Output format ('json' or 'csv')
        """
        output_path = Path(output_path)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        elif format == 'csv':
            # Flatten metrics for CSV
            flat_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        flat_metrics[f"{key}_{self.class_names[i]}"] = v
                else:
                    flat_metrics[key] = value
            
            df = pd.DataFrame([flat_metrics])
            df.to_csv(output_path, index=False)
        
        print(f"✓ Metrics saved to {output_path}")
