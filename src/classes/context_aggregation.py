"""
Context Aggregation Modules for PanCAN

This module contains additional context aggregation components used in PanCAN:
    - Graph-based context modeling
    - Feature importance analysis
    - Interpretability utilities

Reference:
    Jiu et al. "Multi-label Classification with Panoptic Context Aggregation Networks"
    arXiv:2512.23486, December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict
from PIL import Image


class FeatureImportanceAnalyzer:
    """
    Analyze global and local feature importance from PanCAN model.
    
    Global Importance:
    -----------------
    Measures which features contribute most to predictions across the dataset.
    Uses gradient-weighted attention analysis.
    
    Local Importance:
    ----------------
    Explains individual predictions by showing which image regions and scales
    are most important for a specific classification.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Args:
            model: PanCAN model
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
    @torch.enable_grad()
    def compute_gradcam(
        self, 
        image: torch.Tensor, 
        target_class: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Grad-CAM for PanCAN model.
        
        Grad-CAM formula:
        ----------------
        α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A_ij^k)
        L^c = ReLU(Σ_k α_k^c · A^k)
        
        Args:
            image: Input image [1, 3, H, W]
            target_class: Target class (default: predicted class)
        
        Returns:
            Dict with CAMs for each scale
        """
        self.model.eval()
        image = image.to(self.device)
        image.requires_grad = True
        
        # Forward pass with feature retention
        output_dict = self.model(image, return_features=True)
        logits = output_dict['logits']
        scale_features = output_dict['scale_features']
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Backward pass
        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)
        
        # Compute CAMs for each scale
        cams = {}
        for i, features in enumerate(scale_features):
            if features.grad is not None:
                # Gradient-weighted importance
                weights = features.grad.mean(dim=1, keepdim=True)  # [B, 1, D]
                cam = (weights * features).sum(dim=-1)  # [B, N]
                cam = F.relu(cam)
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cams[f'scale_{i+1}'] = cam
        
        return {
            'cams': cams,
            'prediction': target_class,
            'logits': logits.detach()
        }
    
    def compute_attention_importance(
        self, 
        image: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from cross-scale aggregation.
        
        Shows which scales and spatial locations are most important.
        
        Args:
            image: Input image [1, 3, H, W]
        
        Returns:
            Dict with attention analysis
        """
        self.model.eval()
        image = image.to(self.device)
        
        # Get scale weights from cross-scale aggregation
        scale_weights = F.softmax(
            self.model.cross_scale.scale_weights, dim=0
        ).detach().cpu()
        
        # Forward pass
        output_dict = self.model(image, return_features=True)
        
        return {
            'scale_weights': scale_weights,
            'scale_names': [f'Scale {i+1}' for i in range(len(scale_weights))],
            'prediction': output_dict['logits'].argmax(dim=-1).item()
        }
    
    def global_feature_importance(
        self, 
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute global feature importance across dataset.
        
        Aggregates gradient information across many samples to identify
        which features are consistently important.
        
        Args:
            dataloader: Data loader with images
            num_samples: Number of samples to analyze
        
        Returns:
            Dict with global importance scores
        """
        self.model.eval()
        
        scale_importances = [[] for _ in range(4)]  # 4 scales
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
                
            images = batch['pixel_values'] if isinstance(batch, dict) else batch[0]
            images = images.to(self.device)
            
            for img in images:
                if count >= num_samples:
                    break
                    
                result = self.compute_gradcam(img.unsqueeze(0))
                
                for i, (name, cam) in enumerate(result['cams'].items()):
                    scale_importances[i].append(cam.mean().item())
                
                count += 1
        
        return {
            'scale_importance_mean': np.array([np.mean(s) for s in scale_importances]),
            'scale_importance_std': np.array([np.std(s) for s in scale_importances]),
            'num_samples': count
        }


class ContextVisualization:
    """
    Visualization utilities for PanCAN context modeling.
    """
    
    @staticmethod
    def plot_multi_scale_features(
        scale_features: List[torch.Tensor],
        image: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (15, 4)
    ) -> plt.Figure:
        """
        Visualize features at multiple scales.
        
        Args:
            scale_features: List of feature tensors
            image: Original image (optional)
            figsize: Figure size
        
        Returns:
            matplotlib Figure
        """
        num_scales = len(scale_features)
        cols = num_scales + (1 if image is not None else 0)
        
        fig, axes = plt.subplots(1, cols, figsize=figsize)
        
        idx = 0
        if image is not None:
            axes[idx].imshow(image)
            axes[idx].set_title('Original Image')
            axes[idx].axis('off')
            idx += 1
        
        for i, features in enumerate(scale_features):
            # Take mean across feature dimension
            if features.dim() == 3:  # [B, N, D]
                feat_map = features[0].mean(dim=-1).cpu().numpy()
                size = int(np.sqrt(len(feat_map)))
                feat_map = feat_map.reshape(size, size)
            else:
                feat_map = features[0].mean(dim=0).cpu().numpy()
            
            im = axes[idx].imshow(feat_map, cmap='viridis')
            axes[idx].set_title(f'Scale {i+1} ({feat_map.shape[0]}×{feat_map.shape[1]})')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046)
            idx += 1
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_random_walk_visualization(
        affinity_matrix: torch.Tensor,
        transition_matrices: List[torch.Tensor],
        figsize: Tuple[int, int] = (15, 4)
    ) -> plt.Figure:
        """
        Visualize random walk propagation.
        
        Args:
            affinity_matrix: Initial affinity [N, N]
            transition_matrices: List of P^k matrices
            figsize: Figure size
        
        Returns:
            matplotlib Figure
        """
        num_plots = 1 + len(transition_matrices)
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)
        
        # Affinity matrix
        im0 = axes[0].imshow(affinity_matrix.cpu().numpy(), cmap='hot')
        axes[0].set_title('Affinity Matrix A')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        # Transition matrices at different orders
        for i, P_k in enumerate(transition_matrices):
            im = axes[i+1].imshow(P_k.cpu().numpy(), cmap='hot')
            axes[i+1].set_title(f'P^{i+1} (Order {i+1})')
            plt.colorbar(im, ax=axes[i+1], fraction=0.046)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_scale_attention(
        scale_weights: torch.Tensor,
        scale_names: List[str],
        figsize: Tuple[int, int] = (8, 5)
    ) -> plt.Figure:
        """
        Visualize cross-scale attention weights.
        
        Args:
            scale_weights: Attention weights for each scale
            scale_names: Names for each scale
            figsize: Figure size
        
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        weights = scale_weights.cpu().numpy()
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(weights)))
        
        bars = ax.bar(scale_names, weights, color=colors, edgecolor='black')
        
        # Add value labels
        for bar, w in zip(bars, weights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{w:.3f}',
                ha='center', va='bottom', fontsize=11
            )
        
        ax.set_ylabel('Attention Weight')
        ax.set_title('Cross-Scale Attention Distribution')
        ax.set_ylim(0, max(weights) * 1.2)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cam_overlay(
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (10, 4)
    ) -> plt.Figure:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image [H, W, 3]
            cam: Class activation map [H, W]
            alpha: Overlay transparency
            figsize: Figure size
        
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # CAM heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Class Activation Map')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image)
        axes[2].imshow(cam, cmap='jet', alpha=alpha)
        axes[2].set_title('CAM Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig


class GraphContextModule(nn.Module):
    """
    Graph-based context modeling using message passing.
    
    Alternative approach to random walks that uses explicit
    graph neural network operations.
    
    Message Passing:
    ---------------
    h_i^(l+1) = σ(Σ_j∈N(i) (1/√(d_i·d_j)) · W · h_j^(l))
    
    where N(i) is the neighborhood of node i, d_i is degree of node i.
    """
    
    def __init__(
        self, 
        feature_dim: int, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Feature dimension
            num_layers: Number of message passing layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply graph convolution.
        
        Args:
            x: Node features [B, N, D]
            adj: Adjacency matrix [B, N, N] (computed if None)
        
        Returns:
            Updated features [B, N, D]
        """
        if adj is None:
            # Compute adjacency from features
            sim = torch.bmm(x, x.transpose(-2, -1))
            adj = F.softmax(sim, dim=-1)
        
        # Symmetric normalization
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        adj_norm = adj / deg.sqrt() / deg.sqrt().transpose(-2, -1)
        
        # Message passing layers
        h = x
        for layer, norm in zip(self.layers, self.norms):
            # Aggregate neighbors
            h_new = torch.bmm(adj_norm, h)
            h_new = layer(h_new)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            h = h + h_new
        
        return h
