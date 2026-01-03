"""
Model Interpretability Tools for Mission 8
Grad-CAM, SHAP, and Attention Visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import cv2


class GradCAMVisualizer:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    Visualizes which regions of the image are important for predictions
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Navigate to target layer
        target = self.model
        for attr in self.target_layer.split('.'):
            target = getattr(target, attr)
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, image, class_idx=None):
        """
        Generate Class Activation Map
        
        Args:
            image: Input image tensor [1, 3, H, W]
            class_idx: Target class (if None, uses predicted class)
        
        Returns:
            cam: Heatmap [H, W]
            pred_class: Predicted class index
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep positive influences
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input image size
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        
        return cam, class_idx


def plot_gradcam_grid(gradcam, images, labels, image_paths, class_names, save_dir=None):
    """
    Plot grid of original images with Grad-CAM overlays
    
    Args:
        gradcam: GradCAMVisualizer instance
        images: Tensor of images
        labels: True labels
        image_paths: Paths to original images
        class_names: List of class names
        save_dir: Optional save directory
    """
    num_samples = len(images)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img_tensor, label, img_path) in enumerate(zip(images, labels, image_paths)):
        # Original image
        img_np = plt.imread(img_path)
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f'Original\n{class_names[label]}', fontsize=10)
        axes[idx, 0].axis('off')
        
        # Generate Grad-CAM
        img_batch = img_tensor.unsqueeze(0).to(next(gradcam.model.parameters()).device)
        cam, pred_class = gradcam.generate_cam(img_batch)
        
        # Heatmap
        axes[idx, 1].imshow(cam, cmap='jet')
        axes[idx, 1].set_title(f'Grad-CAM Heatmap\nPred: {class_names[pred_class]}', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Overlay
        overlay = cv2.resize(img_np, (cam.shape[1], cam.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Overlay\nConfidence: {pred_class}', fontsize=10)
        axes[idx, 2].axis('off')
    
    plt.suptitle('Grad-CAM Visualization - Model Attention', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'gradcam_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) for feature importance
    """
    
    def __init__(self, model, data_loader, device, num_samples=100):
        """
        Initialize SHAP analyzer
        
        Args:
            model: PyTorch model
            data_loader: DataLoader for background samples
            device: torch device
            num_samples: Number of background samples
        """
        self.model = model
        self.device = device
        
        # Get background samples
        self.background_data = []
        for images, _ in data_loader:
            self.background_data.append(images)
            if len(self.background_data) * images.shape[0] >= num_samples:
                break
        
        self.background_data = torch.cat(self.background_data, dim=0)[:num_samples]
        self.background_data = self.background_data.to(device)
    
    def compute_shap_values(self, num_test_samples=50):
        """
        Compute SHAP values using model predictions
        
        Returns:
            shap_values: Dictionary of SHAP values per class
        """
        import shap
        
        # Define prediction function
        def predict_fn(x):
            x = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
            return F.softmax(outputs, dim=1).cpu().numpy()
        
        # Create explainer
        explainer = shap.KernelExplainer(
            predict_fn,
            self.background_data.cpu().numpy()
        )
        
        # Compute SHAP values
        test_data = self.background_data[:num_test_samples].cpu().numpy()
        shap_values = explainer.shap_values(test_data)
        
        return shap_values


def plot_shap_summary(shap_values, background_data, class_names, save_dir=None):
    """
    Plot SHAP summary with Plotly
    
    Args:
        shap_values: SHAP values array
        background_data: Background samples
        class_names: List of class names
        save_dir: Optional save directory
    """
    import shap
    
    # SHAP summary plot for each class
    fig = plt.figure(figsize=(15, 8))
    
    for class_idx, class_name in enumerate(class_names[:3]):  # Show top 3 classes
        plt.subplot(1, 3, class_idx + 1)
        shap.summary_plot(
            shap_values[class_idx],
            background_data.cpu().numpy(),
            show=False,
            max_display=20
        )
        plt.title(f'SHAP Values - {class_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'shap_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_attention_weights(model, images, class_names, save_dir=None):
    """
    Visualize attention weights from context aggregation module
    
    Args:
        model: PanCANLite model
        images: Input images
        class_names: List of class names
        save_dir: Optional save directory
    """
    model.eval()
    device = next(model.parameters()).device
    images = images.to(device)
    
    # Extract attention weights
    attention_maps = []
    
    def hook_fn(module, input, output):
        # Capture attention weights if module has them
        if hasattr(module, 'attention_weights'):
            attention_maps.append(module.attention_weights.detach())
    
    # Register hooks on context aggregation modules
    hooks = []
    for name, module in model.named_modules():
        if 'context_aggregation' in name.lower() or 'attention' in name.lower():
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Plot attention weights
    if attention_maps:
        fig, axes = plt.subplots(len(images), len(attention_maps), figsize=(15, 5*len(images)))
        
        if len(images) == 1:
            axes = axes.reshape(1, -1)
        
        for img_idx in range(len(images)):
            for att_idx, att_map in enumerate(attention_maps):
                att_viz = att_map[img_idx].cpu().numpy()
                
                if len(att_viz.shape) == 2:
                    axes[img_idx, att_idx].imshow(att_viz, cmap='viridis')
                else:
                    axes[img_idx, att_idx].imshow(att_viz.mean(axis=0), cmap='viridis')
                
                axes[img_idx, att_idx].set_title(f'Attention Map {att_idx+1}', fontsize=10)
                axes[img_idx, att_idx].axis('off')
        
        plt.suptitle('Attention Weight Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'attention_weights.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    else:
        print("⚠️ No attention weights found in model")


if __name__ == "__main__":
    print("Interpretability tools for Mission 8")
    print("Available: GradCAMVisualizer, SHAPAnalyzer, plot_attention_weights")
