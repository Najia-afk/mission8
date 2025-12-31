"""
Visualization Utilities for Mission 8.

Provides plotting functions for data exploration,
model training, and results visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_class_distribution(
    class_counts: pd.Series,
    title: str = "Class Distribution"
) -> go.Figure:
    """
    Plot class distribution as bar chart.
    
    Args:
        class_counts: Series with class counts
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        title=title,
        labels={'x': 'Category', 'y': 'Count'},
        color=class_counts.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def plot_sample_images(
    image_paths: List[Path],
    labels: List[str],
    n_cols: int = 5,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot a grid of sample images.
    
    Args:
        image_paths: List of image file paths
        labels: List of labels for each image
        n_cols: Number of columns
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        try:
            img = Image.open(path)
            axes[idx].imshow(img)
            axes[idx].set_title(label, fontsize=10)
        except Exception as e:
            axes[idx].text(0.5, 0.5, 'Error', ha='center', va='center')
        
        axes[idx].axis('off')
    
    # Hide unused axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict,
    title: str = "Training History"
) -> go.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with loss and accuracy history
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Loss', 'Accuracy']
    )
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Loss
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['train_loss'],
            name='Train Loss',
            mode='lines',
            line=dict(color='#3498db')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['val_loss'],
            name='Val Loss',
            mode='lines',
            line=dict(color='#e74c3c', dash='dash')
        ),
        row=1, col=1
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['train_acc'],
            name='Train Acc',
            mode='lines',
            line=dict(color='#3498db'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history['val_acc'],
            name='Val Acc',
            mode='lines',
            line=dict(color='#e74c3c', dash='dash'),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=400
    )
    
    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=2)
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = True
) -> go.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize
        
    Returns:
        Plotly figure
    """
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Create text annotations
    text = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm)):
            if normalize:
                row.append(f'{cm[i,j]}<br>({cm_display[i,j]:.2f})')
            else:
                row.append(str(cm[i,j]))
        text.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=text,
        texttemplate='%{text}',
        textfont={'size': 10},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='True',
        height=500,
        width=600
    )
    
    return fig


def plot_metrics_radar(
    metrics_dict: Dict[str, Dict],
    metric_names: List[str] = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
) -> go.Figure:
    """
    Plot radar chart comparing model metrics.
    
    Args:
        metrics_dict: Dictionary of model metrics
        metric_names: List of metrics to plot
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_names + [metric_names[0]],
            fill='toself',
            name=model_name,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Model Comparison - Radar Chart',
        showlegend=True
    )
    
    return fig


def plot_per_class_performance(
    per_class_metrics: Dict[str, List[float]],
    class_names: List[str],
    metric_name: str = "F1 Score"
) -> go.Figure:
    """
    Plot per-class performance comparison.
    
    Args:
        per_class_metrics: Dictionary mapping model name to per-class scores
        class_names: List of class names
        metric_name: Name of the metric
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (model_name, scores) in enumerate(per_class_metrics.items()):
        fig.add_trace(go.Bar(
            name=model_name,
            x=class_names,
            y=scores,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title=f'Per-Class {metric_name} Comparison',
        xaxis_title='Class',
        yaxis_title=metric_name,
        barmode='group',
        xaxis_tickangle=-45
    )
    
    return fig


def save_figure(
    fig,
    path: Path,
    format: str = 'png',
    width: int = 1200,
    height: int = 800
):
    """
    Save a figure to file.
    
    Args:
        fig: Matplotlib or Plotly figure
        path: Output path
        format: Output format
        width: Width in pixels (for Plotly)
        height: Height in pixels (for Plotly)
    """
    path = Path(path)
    
    if hasattr(fig, 'write_image'):  # Plotly
        fig.write_image(str(path), width=width, height=height)
    else:  # Matplotlib
        fig.savefig(path, dpi=150, bbox_inches='tight')
    
    print(f"✓ Figure saved to {path}")


def create_comparison_dashboard(
    model_results: Dict,
    class_names: List[str]
) -> go.Figure:
    """
    Create a comprehensive comparison dashboard.
    
    Args:
        model_results: Dictionary of model results
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Overall Metrics',
            'Per-Class F1 Scores',
            'Confusion Matrix (Model 1)',
            'Confusion Matrix (Model 2)'
        ],
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'heatmap'}, {'type': 'heatmap'}]
        ]
    )
    
    # Add traces here...
    # This is a template - actual implementation depends on data structure
    
    fig.update_layout(
        title='Model Comparison Dashboard',
        height=800,
        showlegend=True
    )
    
    return fig
