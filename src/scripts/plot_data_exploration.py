"""
Data Exploration Visualizations for Mission 8
Plots class distribution and sample images from the Flipkart dataset
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_class_distribution(train_counts, save_dir=None):
    """
    Plot training set class distribution
    
    Args:
        train_counts: Dictionary of class names to counts
        save_dir: Optional directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    classes = list(train_counts.keys())
    counts = list(train_counts.values())

    bars = ax.bar(classes, counts, color='steelblue', edgecolor='black')
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Training Set Class Distribution', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', fontsize=10)

    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'class_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

    balance_status = 'Balanced' if max(counts) - min(counts) < 10 else 'Imbalanced'
    print(f"\nClass balance: {balance_status}")


def plot_sample_images(data_loader, train_loader, save_dir=None):
    """
    Plot original and augmented sample images
    
    Args:
        data_loader: FlipkartDataLoader instance
        train_loader: PyTorch DataLoader for training
        save_dir: Optional directory to save plot
    """
    fig, axes = plt.subplots(2, 7, figsize=(16, 5))

    # Get one sample per class (original)
    for idx, class_name in enumerate(data_loader.class_names):
        for i, label in enumerate(data_loader.train_dataset.labels):
            if label == idx:
                img_path = data_loader.train_dataset.images[i]
                img = plt.imread(img_path)
                axes[0, idx].imshow(img)
                axes[0, idx].set_title(class_name[:15], fontsize=9)
                axes[0, idx].axis('off')
                break

    # Get augmented versions
    batch_images, batch_labels = next(iter(train_loader))
    for idx in range(min(7, len(batch_images))):
        img = batch_images[idx].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axes[1, idx].imshow(img)
        axes[1, idx].set_title(f"Aug: {data_loader.class_names[batch_labels[idx]]}"[:15], fontsize=9)
        axes[1, idx].axis('off')

    plt.suptitle('Original (top) vs Augmented (bottom) Images', fontsize=14)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'sample_images.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Data exploration plotting utilities for Mission 8")
    print("Import and use: plot_class_distribution(), plot_sample_images()")
