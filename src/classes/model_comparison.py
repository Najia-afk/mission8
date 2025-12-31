"""Model Comparison Module for PanCAN vs VGG16.

This module provides utilities to compare the performance
of PanCAN and VGG16 models on the same dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ModelComparison:
    """
    Compare performance of multiple models.
    
    Provides utilities for computing metrics, generating
    visualizations, and creating comparison reports.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize the comparison module.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.results = {}
        
    def add_model_results(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        training_time: Optional[float] = None,
        num_params: Optional[int] = None
    ):
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            training_time: Training time in seconds (optional)
            num_params: Number of model parameters (optional)
        """
        metrics = self._compute_metrics(y_true, y_pred)
        
        self.results[model_name] = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics,
            'training_time': training_time,
            'num_params': num_params
        }
        
        print(f"✓ Added results for {model_name}")
        
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro'),
            'per_class_f1': f1_score(y_true, y_pred, average=None),
        }
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table.
        
        Returns:
            DataFrame with model comparison
        """
        data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Macro F1': f"{metrics['macro_f1']:.4f}",
                'Weighted F1': f"{metrics['weighted_f1']:.4f}",
                'Macro Precision': f"{metrics['macro_precision']:.4f}",
                'Macro Recall': f"{metrics['macro_recall']:.4f}",
            }
            
            if result['training_time']:
                row['Training Time'] = f"{result['training_time']:.1f}s"
            
            if result['num_params']:
                row['Parameters'] = f"{result['num_params']:,}"
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_metrics_comparison(
        self,
        metrics: List[str] = ['accuracy', 'macro_f1', 'weighted_f1']
    ) -> go.Figure:
        """
        Create bar chart comparing metrics.
        
        Args:
            metrics: List of metrics to compare
            
        Returns:
            Plotly figure
        """
        model_names = list(self.results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        fig = go.Figure()
        
        for i, metric in enumerate(metrics):
            values = [
                self.results[model]['metrics'].get(metric, 0)
                for model in model_names
            ]
            
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=model_names,
                y=values,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def plot_confusion_matrices(self) -> plt.Figure:
        """
        Plot confusion matrices for all models.
        
        Returns:
            Matplotlib figure
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (model_name, result) in zip(axes, self.results.items()):
            cm = confusion_matrix(result['y_true'], result['y_pred'])
            
            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
            ax.set_title(f'{model_name}\nConfusion Matrix')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Labels
            tick_marks = np.arange(self.num_classes)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_yticklabels(self.class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                           ha='center', va='center',
                           color='white' if cm_normalized[i, j] > thresh else 'black',
                           fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_per_class_f1(self) -> go.Figure:
        """
        Plot per-class F1 scores for all models.
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for i, (model_name, result) in enumerate(self.results.items()):
            per_class_f1 = result['metrics']['per_class_f1']
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=self.class_names,
                y=per_class_f1,
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title='Per-Class F1 Score Comparison',
            xaxis_title='Class',
            yaxis_title='F1 Score',
            barmode='group',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def get_classification_reports(self) -> Dict[str, str]:
        """
        Generate classification reports for all models.
        
        Returns:
            Dictionary of classification report strings
        """
        reports = {}
        
        for model_name, result in self.results.items():
            report = classification_report(
                result['y_true'],
                result['y_pred'],
                target_names=self.class_names,
                digits=4
            )
            reports[model_name] = report
        
        return reports
    
    def plot_training_curves(
        self,
        histories: Dict[str, Dict]
    ) -> go.Figure:
        """
        Plot training curves for all models.
        
        Args:
            histories: Dictionary of training histories
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Loss', 'Accuracy']
        )
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = colors[i % len(colors)]
            
            # Loss
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['train_loss'],
                    name=f'{model_name} (train)',
                    line=dict(color=color),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val_loss'],
                    name=f'{model_name} (val)',
                    line=dict(color=color, dash='dash'),
                    mode='lines'
                ),
                row=1, col=1
            )
            
            # Accuracy
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['train_acc'],
                    name=f'{model_name} (train)',
                    line=dict(color=color),
                    mode='lines',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val_acc'],
                    name=f'{model_name} (val)',
                    line=dict(color=color, dash='dash'),
                    mode='lines',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Training Curves Comparison',
            height=400
        )
        
        fig.update_xaxes(title_text='Epoch', row=1, col=1)
        fig.update_xaxes(title_text='Epoch', row=1, col=2)
        fig.update_yaxes(title_text='Loss', row=1, col=1)
        fig.update_yaxes(title_text='Accuracy', row=1, col=2)
        
        return fig
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report.
        
        Returns:
            Summary report string
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON SUMMARY")
        report.append("=" * 60)
        
        # Find best model for each metric
        metrics_of_interest = ['accuracy', 'macro_f1', 'weighted_f1']
        
        for metric in metrics_of_interest:
            best_model = max(
                self.results.keys(),
                key=lambda x: self.results[x]['metrics'][metric]
            )
            best_value = self.results[best_model]['metrics'][metric]
            
            report.append(f"\nBest {metric.replace('_', ' ').title()}: {best_model} ({best_value:.4f})")
        
        report.append("\n" + "-" * 60)
        report.append("DETAILED METRICS")
        report.append("-" * 60)
        
        for model_name, result in self.results.items():
            report.append(f"\n{model_name}:")
            for metric, value in result['metrics'].items():
                if metric != 'per_class_f1':
                    report.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: Path):
        """
        Save all comparison results.
        
        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        df = self.get_comparison_table()
        df.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        # Save summary report
        report = self.generate_summary_report()
        with open(output_dir / 'comparison_summary.txt', 'w') as f:
            f.write(report)
        
        # Save classification reports
        reports = self.get_classification_reports()
        for model_name, report in reports.items():
            safe_name = model_name.replace(' ', '_').lower()
            with open(output_dir / f'{safe_name}_classification_report.txt', 'w') as f:
                f.write(report)
        
        print(f"✓ Results saved to {output_dir}")
