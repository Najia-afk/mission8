"""
Saliency Map Visualization for PanCANLite and ViT models.

References:
- [Simonyan et al., 2014] "Deep Inside Convolutional Networks: Visualising Image 
  Classification Models and Saliency Maps"
- [Selvaraju et al., 2017] "Grad-CAM: Visual Explanations from Deep Networks"
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path


class SaliencyMapVisualizer:
    """
    Input Gradient Saliency Maps for neural network architectures.
    Works with both CNN (PanCANLite) and Transformer (ViT) models.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def generate_saliency(self, input_tensor, class_idx):
        """Generate saliency map using input gradients."""
        self.model.eval()
        
        # Create input that requires gradients
        input_img = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_img)
        
        # Backward pass for target class
        self.model.zero_grad()
        if input_img.grad is not None:
            input_img.grad.zero_()
        
        output[0, class_idx].backward()
        
        # Get gradients w.r.t. input
        if input_img.grad is None:
            print("⚠️ Warning: No gradients computed - check gradient flow")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        saliency = input_img.grad.data.abs()
        
        # Take max across color channels
        saliency, _ = saliency.max(dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        
        # Normalize
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency


def collect_class_samples(test_loader, num_classes, samples_per_class=1):
    """Collect one sample per class from test loader."""
    class_samples = {i: [] for i in range(num_classes)}
    
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            label_idx = label.item()
            if len(class_samples[label_idx]) < samples_per_class:
                class_samples[label_idx].append((img, label_idx))
        if all(len(samples) >= samples_per_class for samples in class_samples.values()):
            break
    
    return class_samples


def plot_saliency_maps(model, test_loader, class_names, device, 
                       save_dir=None, model_name="Model", num_samples=5,
                       title=None, save_path=None):
    """
    Generate and plot saliency map visualizations for a model.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        class_names: List of class names
        device: torch device
        save_dir: Directory to save the plot (optional, deprecated - use save_path)
        model_name: Name for the plot title (deprecated - use title)
        num_samples: Number of samples to visualize
        title: Title for the plot (preferred over model_name)
        save_path: Path to save the plot (preferred over save_dir)
    
    Returns:
        fig: matplotlib figure
    """
    # Handle parameter aliases
    if title is not None:
        model_name = title
    if save_path is not None:
        save_dir = save_path
    
    print(f"📊 Generating {model_name} Saliency Map visualizations...")
    
    # Initialize visualizer
    saliency_viz = SaliencyMapVisualizer(model, device)
    
    # Collect samples
    num_classes = len(class_names)
    class_samples = collect_class_samples(test_loader, num_classes)
    
    # Generate visualizations
    num_samples = min(num_samples, len(class_samples))
    fig = plt.figure(figsize=(20, 4 * num_samples))
    plt.subplots_adjust(hspace=0.4)
    
    for idx in range(num_samples):
        if len(class_samples[idx]) == 0:
            continue
        
        img_tensor, true_label = class_samples[idx][0]
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(img_batch)
        probs = F.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
        pred_confidence = probs[pred_class].item()
        
        # Generate Saliency Map
        saliency = saliency_viz.generate_saliency(img_batch, pred_class)
        
        # Prepare original image for display
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Resize saliency to image size
        saliency_resized = cv2.resize(saliency, (img_np.shape[1], img_np.shape[0]))
        
        # Apply Gaussian blur for smoother visualization
        saliency_smooth = cv2.GaussianBlur(saliency_resized, (11, 11), 0)
        saliency_smooth = saliency_smooth / (saliency_smooth.max() + 1e-8)
        
        # Create colored heatmap (RGB only)
        heatmap_colored = cm.jet(saliency_smooth)[:, :, :3]
        
        # Create overlay
        overlay = 0.6 * img_np + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        # Plot 4 panels per row
        ax1 = plt.subplot(num_samples, 4, idx * 4 + 1)
        ax1.imshow(img_np)
        ax1.set_title(f"Original: {class_names[true_label]}", fontsize=12)
        ax1.axis('off')
        
        ax2 = plt.subplot(num_samples, 4, idx * 4 + 2)
        ax2.imshow(saliency_smooth, cmap='jet')
        ax2.set_title("Saliency Map", fontsize=12)
        ax2.axis('off')
        
        ax3 = plt.subplot(num_samples, 4, idx * 4 + 3)
        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=12)
        ax3.axis('off')
        
        ax4 = plt.subplot(num_samples, 4, idx * 4 + 4)
        ax4.axis('off')
        top3_probs, top3_idxs = torch.topk(probs, 3)
        text_str = f"True: {class_names[true_label]}\nPred: {class_names[pred_class]}\nConf: {pred_confidence:.2%}\n\nTop 3:\n"
        for i in range(3):
            text_str += f"{i+1}. {class_names[top3_idxs[i].item()]}: {top3_probs[i].item():.1%}\n"
        status_color = 'green' if true_label == pred_class else 'red'
        ax4.text(0.0, 0.5, text_str, fontsize=12, va='center', fontfamily='monospace',
                 bbox=dict(facecolor=status_color, alpha=0.1, boxstyle='round,pad=1'))
    
    plt.suptitle(f"{model_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save if path provided
    if save_dir:
        # If save_dir is a full file path (has extension), use it directly
        save_dir_path = Path(save_dir)
        if save_dir_path.suffix in ['.png', '.jpg', '.pdf', '.svg']:
            final_save_path = save_dir_path
        else:
            # It's a directory, construct the filename
            final_save_path = save_dir_path / f'{model_name.lower().replace(" ", "_").replace("-", "_")}_saliency_maps.png'
        
        # Ensure parent directory exists
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(final_save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {final_save_path}")
    
    plt.show()
    print(f"✅ {model_name} Saliency visualization complete.")
    
    return fig
