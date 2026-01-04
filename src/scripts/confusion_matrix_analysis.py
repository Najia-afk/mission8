"""
Confusion Matrix and Classification Metrics Analysis.

Generates interactive Plotly confusion matrix and per-class metrics visualizations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix_plotly(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Create an interactive Plotly confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
    
    Returns:
        fig: Plotly figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create Plotly heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{text}<br>Rate: %{z:.1%}<extra></extra>',
        colorbar=dict(title="Rate")
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br>" +
                 "<sub>Values show counts, colors show normalized rates</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(title="<b>Predicted Label</b>", side='bottom'),
        yaxis=dict(title="<b>True Label</b>", autorange='reversed'),
        height=700,
        width=800,
        margin=dict(t=100, b=100, l=150, r=100)
    )
    
    return fig


def plot_per_class_metrics_plotly(y_true, y_pred, class_names):
    """
    Create an interactive Plotly bar chart for per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        fig: Plotly figure
    """
    # Get classification report as dict
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names,
                                   output_dict=True)
    
    # Create per-class metrics dataframe
    metrics_data = []
    for class_name in class_names:
        metrics_data.append({
            'Class': class_name,
            'Precision': report[class_name]['precision'],
            'Recall': report[class_name]['recall'],
            'F1-Score': report[class_name]['f1-score'],
            'Support': report[class_name]['support']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Precision',
        x=metrics_df['Class'],
        y=metrics_df['Precision'],
        text=[f"{v:.1%}" for v in metrics_df['Precision']],
        textposition='outside',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        name='Recall',
        x=metrics_df['Class'],
        y=metrics_df['Recall'],
        text=[f"{v:.1%}" for v in metrics_df['Recall']],
        textposition='outside',
        marker_color='rgb(26, 118, 255)'
    ))
    
    fig.add_trace(go.Bar(
        name='F1-Score',
        x=metrics_df['Class'],
        y=metrics_df['F1-Score'],
        text=[f"{v:.1%}" for v in metrics_df['F1-Score']],
        textposition='outside',
        marker_color='rgb(50, 171, 96)'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Per-Class Performance Metrics</b><br>" +
                 "<sub>Precision, Recall, and F1-Score for each product category</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(title="<b>Product Category</b>"),
        yaxis=dict(title="<b>Score</b>", range=[0, 1.1]),
        barmode='group',
        height=500,
        margin=dict(t=100, b=100),
        legend=dict(x=0.85, y=1, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def analyze_confusion_matrix(y_true, y_pred, class_names, overall_acc=None, overall_f1=None):
    """
    Complete confusion matrix analysis with visualizations and printed report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        overall_acc: Overall accuracy (optional, will compute if None)
        overall_f1: Overall F1 score (optional, will compute if None)
    
    Returns:
        tuple: (confusion_matrix_fig, metrics_fig)
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    print("📊 Computing confusion matrix and per-class metrics...")
    
    # Compute overall metrics if not provided
    if overall_acc is None:
        overall_acc = accuracy_score(y_true, y_pred)
    if overall_f1 is None:
        overall_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Create visualizations
    fig_cm = plot_confusion_matrix_plotly(y_true, y_pred, class_names, 
                                          title="Confusion Matrix - Model Performance")
    fig_cm.show()
    
    fig_metrics = plot_per_class_metrics_plotly(y_true, y_pred, class_names)
    fig_metrics.show()
    
    # Print detailed report
    print("\n" + "="*70)
    print("PER-CLASS PERFORMANCE METRICS")
    print("="*70)
    print(classification_report(y_true, y_pred, 
                               target_names=class_names,
                               digits=4))
    
    print(f"\n✅ Confusion matrix and per-class analysis complete")
    print(f"📊 Overall Accuracy: {overall_acc:.4f} ({100*overall_acc:.2f}%)")
    print(f"📊 Macro F1-Score: {overall_f1:.4f} ({100*overall_f1:.2f}%)")
    
    return fig_cm, fig_metrics
