"""
Training Curves Visualization for Mission 8
Plots training history with both matplotlib and interactive plotly
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def plot_training_curves_matplotlib(history, save_dir=None):
    """
    Plot training curves using matplotlib
    
    Args:
        history: Dictionary with training history (train_loss, val_loss, train_acc, val_acc, lr)
        save_dir: Optional directory to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    axes[0].plot(history['val_loss'], label='Val', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot([100*x for x in history['train_acc']], label='Train', color='blue')
    axes[1].plot([100*x for x in history['val_acc']], label='Val', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[2].plot(history['lr'], color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_training_curves_plotly(history, save_dir=None):
    """
    Plot interactive training curves using Plotly
    
    Args:
        history: Dictionary with training history
        save_dir: Optional directory to save HTML
    """
    # Create subplot figure with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy', 'Learning Rate Schedule'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # 1. Loss curves
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['train_loss'])+1)),
                   y=history['train_loss'],
                   mode='lines+markers',
                   name='Train Loss',
                   line=dict(color='#3498db', width=2),
                   marker=dict(size=6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['val_loss'])+1)),
                   y=history['val_loss'],
                   mode='lines+markers',
                   name='Val Loss',
                   line=dict(color='#e74c3c', width=2),
                   marker=dict(size=6)),
        row=1, col=1
    )

    # 2. Accuracy curves
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['train_acc'])+1)),
                   y=[100*x for x in history['train_acc']],
                   mode='lines+markers',
                   name='Train Accuracy',
                   line=dict(color='#2ecc71', width=2),
                   marker=dict(size=6)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['val_acc'])+1)),
                   y=[100*x for x in history['val_acc']],
                   mode='lines+markers',
                   name='Val Accuracy',
                   line=dict(color='#f39c12', width=2),
                   marker=dict(size=6)),
        row=2, col=1
    )

    # 3. Learning rate
    fig.add_trace(
        go.Scatter(x=list(range(1, len(history['lr'])+1)),
                   y=history['lr'],
                   mode='lines+markers',
                   name='Learning Rate',
                   line=dict(color='#9b59b6', width=2),
                   marker=dict(size=6)),
        row=3, col=1
    )

    # Update layout
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)

    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Learning Rate", row=3, col=1, type="log")

    fig.update_layout(
        height=1000,
        title_text="PanCANLite Training History - Interactive View",
        title_font_size=18,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )

    if save_dir:
        save_path = Path(save_dir) / 'training_curves_interactive.html'
        fig.write_html(str(save_path))
        print(f"Saved interactive plot to {save_path}")

    fig.show()

    # Print training summary
    best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
    best_val_acc = max(history['val_acc'])
    print(f"\n📊 Training Summary:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Best Val Accuracy: {100*best_val_acc:.2f}%")
    print(f"  Final Train Accuracy: {100*history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Accuracy: {100*history['val_acc'][-1]:.2f}%")


if __name__ == "__main__":
    print("Training curves plotting utilities for Mission 8")
    print("Import and use: plot_training_curves_matplotlib(), plot_training_curves_plotly()")
