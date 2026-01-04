"""
Model Interpretability Tools for Mission 8
Grad-CAM, SHAP, Saliency Maps, and Attention Visualization

This module provides comprehensive interpretability tools for deep learning models,
supporting both CNN and Vision Transformer architectures.

References:
    - Grad-CAM: [Selvaraju et al., 2017] "Grad-CAM: Visual Explanations from Deep Networks"
    - SHAP: [Lundberg & Lee, 2017] "A Unified Approach to Interpreting Model Predictions"
    - Saliency Maps: [Simonyan et al., 2014] "Deep Inside Convolutional Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional, List, Tuple, Union
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
    print("           SaliencyMapVisualizer, ViTSaliencyVisualizer")


# =============================================================================
# SALIENCY MAP VISUALIZERS
# =============================================================================

class SaliencyMapVisualizer:
    """
    Input Gradient Saliency Maps for CNN architectures (e.g., PanCANLite, VGG16).
    
    Saliency maps highlight which input pixels have the highest gradient with
    respect to the predicted class, indicating regions important for the prediction.
    
    Works with fixed gradient flow in EfficientGridFeatureExtractor.
    
    Example:
        >>> visualizer = SaliencyMapVisualizer(model)
        >>> saliency = visualizer.generate_saliency(image_batch, class_idx=0)
        >>> visualizer.plot_saliency(image_batch, saliency, class_names)
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize saliency map visualizer.
        
        Args:
            model: PyTorch model (CNN-based)
        """
        self.model = model
    
    def generate_saliency(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: int
    ) -> np.ndarray:
        """
        Generate saliency map using input gradients.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class index for gradient computation
            
        Returns:
            Normalized saliency map [H, W]
        """
        self.model.eval()
        
        # Create input that requires gradients
        input_img = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass - gradients flow through backbone
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


class ViTSaliencyVisualizer:
    """
    Input Gradient Saliency Maps for Vision Transformer models.
    
    ViT uses patch-based attention which creates different saliency patterns
    compared to CNN convolutions. This visualizer captures these patterns.
    
    Example:
        >>> visualizer = ViTSaliencyVisualizer(vit_model)
        >>> saliency = visualizer.generate_saliency(image_batch, class_idx=0)
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize ViT saliency map visualizer.
        
        Args:
            model: Vision Transformer model
        """
        self.model = model
    
    def generate_saliency(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: int
    ) -> np.ndarray:
        """
        Generate saliency map using input gradients for ViT.
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Target class index for gradient computation
            
        Returns:
            Normalized saliency map [H, W]
        """
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
            print("⚠️ Warning: No gradients computed")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        saliency = input_img.grad.data.abs()
        saliency, _ = saliency.max(dim=1)
        saliency = saliency.squeeze().cpu().numpy()
        
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        return saliency


def plot_saliency_comparison(
    model_cnn: nn.Module,
    model_vit: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
    num_samples: int = 5,
    save_dir: Optional[Path] = None
) -> None:
    """
    Generate side-by-side saliency comparison: CNN vs ViT.
    
    Args:
        model_cnn: CNN model (PanCANLite or VGG16)
        model_vit: Vision Transformer model
        test_loader: Test data loader
        class_names: List of class names
        device: Device for inference
        num_samples: Number of samples to visualize
        save_dir: Optional directory to save figure
    """
    cnn_viz = SaliencyMapVisualizer(model_cnn)
    vit_viz = ViTSaliencyVisualizer(model_vit)
    
    # Collect one sample per class
    class_samples = {i: [] for i in range(len(class_names))}
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            label_idx = label.item()
            if len(class_samples[label_idx]) < 1:
                class_samples[label_idx].append((img, label_idx))
        if all(len(samples) >= 1 for samples in class_samples.values()):
            break
    
    # Generate visualizations
    fig = plt.figure(figsize=(24, 4 * num_samples))
    plt.subplots_adjust(hspace=0.4)
    
    for idx in range(min(num_samples, len(class_names))):
        if len(class_samples[idx]) == 0:
            continue
        
        img_tensor, true_label = class_samples[idx][0]
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            cnn_output = model_cnn(img_batch)
            vit_output = model_vit(img_batch)
        
        cnn_pred = cnn_output.argmax(dim=1).item()
        vit_pred = vit_output.argmax(dim=1).item()
        
        # Generate saliency maps
        cnn_saliency = cnn_viz.generate_saliency(img_batch, cnn_pred)
        vit_saliency = vit_viz.generate_saliency(img_batch, vit_pred)
        
        # Prepare original image
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Resize and smooth saliency maps
        cnn_saliency_resized = cv2.resize(cnn_saliency, (img_np.shape[1], img_np.shape[0]))
        vit_saliency_resized = cv2.resize(vit_saliency, (img_np.shape[1], img_np.shape[0]))
        
        cnn_saliency_smooth = cv2.GaussianBlur(cnn_saliency_resized, (11, 11), 0)
        vit_saliency_smooth = cv2.GaussianBlur(vit_saliency_resized, (11, 11), 0)
        
        cnn_saliency_smooth = cnn_saliency_smooth / (cnn_saliency_smooth.max() + 1e-8)
        vit_saliency_smooth = vit_saliency_smooth / (vit_saliency_smooth.max() + 1e-8)
        
        # Create overlays
        cnn_heatmap = cm.jet(cnn_saliency_smooth)[:, :, :3]
        vit_heatmap = cm.jet(vit_saliency_smooth)[:, :, :3]
        
        cnn_overlay = 0.6 * img_np + 0.4 * cnn_heatmap
        vit_overlay = 0.6 * img_np + 0.4 * vit_heatmap
        
        cnn_overlay = np.clip(cnn_overlay, 0, 1)
        vit_overlay = np.clip(vit_overlay, 0, 1)
        
        # Plot 6 panels per row
        ax1 = plt.subplot(num_samples, 6, idx * 6 + 1)
        ax1.imshow(img_np)
        ax1.set_title(f"Original: {class_names[true_label]}", fontsize=10)
        ax1.axis('off')
        
        ax2 = plt.subplot(num_samples, 6, idx * 6 + 2)
        ax2.imshow(cnn_saliency_smooth, cmap='jet')
        ax2.set_title("CNN Saliency", fontsize=10)
        ax2.axis('off')
        
        ax3 = plt.subplot(num_samples, 6, idx * 6 + 3)
        ax3.imshow(cnn_overlay)
        ax3.set_title(f"CNN: {class_names[cnn_pred]}", fontsize=10)
        ax3.axis('off')
        
        ax4 = plt.subplot(num_samples, 6, idx * 6 + 4)
        ax4.imshow(vit_saliency_smooth, cmap='jet')
        ax4.set_title("ViT Saliency", fontsize=10)
        ax4.axis('off')
        
        ax5 = plt.subplot(num_samples, 6, idx * 6 + 5)
        ax5.imshow(vit_overlay)
        ax5.set_title(f"ViT: {class_names[vit_pred]}", fontsize=10)
        ax5.axis('off')
        
        ax6 = plt.subplot(num_samples, 6, idx * 6 + 6)
        ax6.axis('off')
        status = "✅" if true_label == cnn_pred == vit_pred else "⚠️"
        ax6.text(0.1, 0.5, f"{status} True: {class_names[true_label]}\n"
                          f"CNN: {class_names[cnn_pred]}\n"
                          f"ViT: {class_names[vit_pred]}",
                fontsize=11, va='center', fontfamily='monospace')
    
    plt.suptitle("Saliency Map Comparison: CNN vs Vision Transformer", fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'saliency_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    plt.show()
