"""
PanCAN: Panoptic Context Aggregation Networks for Image Classification

Implementation based on:
    Jiu et al. "Multi-label Classification with Panoptic Context Aggregation Networks"
    arXiv:2512.23486, December 2025

This module implements the core PanCAN architecture:
    1. Multi-scale feature extraction (from CvT backbone)
    2. Multi-order random walks for neighborhood modeling
    3. Cross-scale context aggregation with attention
    4. Classification head

Author: Mission 8 - OpenClassrooms Data Scientist Pathway
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import timm


class MultiOrderRandomWalk(nn.Module):
    """
    Multi-Order Random Walk module for capturing neighborhood relationships.
    
    Mathematical Formulation:
    ------------------------
    Given a feature matrix X ∈ ℝ^(N×D) where N is the number of spatial locations
    and D is the feature dimension:
    
    1. Compute affinity matrix A:
       A_ij = exp(-||x_i - x_j||² / σ²)
    
    2. Normalize to get transition probability matrix P:
       P = D^(-1) · A  where D_ii = Σ_j A_ij
    
    3. Multi-order random walk features:
       Z^(k) = P^k · X  for k = 1, 2, ..., K
    
    This captures k-hop neighborhood information at each order.
    """
    
    def __init__(
        self, 
        feature_dim: int,
        num_orders: int = 3,
        sigma: float = 1.0,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features
            num_orders: Number of random walk orders (K)
            sigma: Temperature for affinity computation
            dropout: Dropout rate
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_orders = num_orders
        self.sigma = sigma
        
        # Learnable projection for each order
        self.order_projections = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) 
            for _ in range(num_orders)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * num_orders, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def compute_affinity_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute affinity matrix using Gaussian kernel.
        
        A_ij = exp(-||x_i - x_j||² / (2σ²))
        
        Args:
            x: Feature tensor [B, N, D]
        
        Returns:
            Affinity matrix [B, N, N]
        """
        # Pairwise squared distances
        # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2·x_i·x_j
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        dist_sq = x_norm + x_norm.transpose(-2, -1) - 2 * torch.bmm(x, x.transpose(-2, -1))
        
        # Gaussian affinity
        affinity = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        
        return affinity
    
    def compute_transition_matrix(self, affinity: torch.Tensor) -> torch.Tensor:
        """
        Normalize affinity to get row-stochastic transition matrix.
        
        P = D^(-1) · A
        
        Args:
            affinity: Affinity matrix [B, N, N]
        
        Returns:
            Transition probability matrix [B, N, N]
        """
        # Row normalization (each row sums to 1)
        row_sum = affinity.sum(dim=-1, keepdim=True) + 1e-8
        transition = affinity / row_sum
        
        return transition
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-order random walks.
        
        Args:
            x: Input features [B, N, D]
        
        Returns:
            Aggregated multi-order features [B, N, D]
        """
        B, N, D = x.shape
        
        # Compute transition matrix
        affinity = self.compute_affinity_matrix(x)
        P = self.compute_transition_matrix(affinity)
        
        # Multi-order random walks
        order_features = []
        P_k = P.clone()  # P^1
        
        for k in range(self.num_orders):
            # Z^(k) = P^k · X
            z_k = torch.bmm(P_k, x)  # [B, N, D]
            z_k = self.order_projections[k](z_k)
            order_features.append(z_k)
            
            # P^(k+1) = P^k · P
            if k < self.num_orders - 1:
                P_k = torch.bmm(P_k, P)
        
        # Concatenate and fuse
        multi_order = torch.cat(order_features, dim=-1)  # [B, N, D*K]
        output = self.fusion(multi_order)  # [B, N, D]
        
        return output


class ScaleAttention(nn.Module):
    """
    Attention mechanism for weighting features within a scale.
    
    Mathematical Formulation:
    ------------------------
    Given features X ∈ ℝ^(N×D):
    
    1. Compute attention scores:
       e_i = w^T · tanh(W · x_i + b)
    
    2. Normalize with softmax:
       α_i = softmax(e_i) = exp(e_i) / Σ_j exp(e_j)
    
    3. Weighted aggregation:
       z = Σ_i α_i · x_i
    """
    
    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention (default: feature_dim // 4)
        """
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 4
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted features.
        
        Args:
            x: Input features [B, N, D]
            mask: Optional mask [B, N]
        
        Returns:
            Tuple of (weighted features [B, D], attention weights [B, N])
        """
        # Attention scores
        scores = self.attention(x).squeeze(-1)  # [B, N]
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax normalization
        weights = F.softmax(scores, dim=-1)  # [B, N]
        
        # Weighted sum
        output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        
        return output, weights


class CrossScaleAggregation(nn.Module):
    """
    Cross-Scale Context Aggregation module.
    
    Mathematical Formulation:
    ------------------------
    Given features from S scales: {F_1, F_2, ..., F_S}
    
    1. Select salient anchors at finer scale using attention
    2. Map features to Hilbert space: φ(F_s)
    3. Aggregate with learned weights:
       F_fused = Σ_s w_s · φ_s(F_s)
    
    The Hilbert space mapping uses Random Fourier Features:
       φ(x) = √(2/D) · cos(Wx + b)
    
    where W ~ N(0, σ²I) and b ~ Uniform(0, 2π)
    """
    
    def __init__(
        self, 
        feature_dims: List[int],
        output_dim: int,
        num_rff_features: int = 512,
        sigma: float = 1.0
    ):
        """
        Args:
            feature_dims: Feature dimensions for each scale
            output_dim: Output feature dimension
            num_rff_features: Number of Random Fourier Features
            sigma: Kernel bandwidth for RFF
        """
        super().__init__()
        self.num_scales = len(feature_dims)
        self.output_dim = output_dim
        self.num_rff = num_rff_features
        
        # Attention for each scale
        self.scale_attentions = nn.ModuleList([
            ScaleAttention(dim) for dim in feature_dims
        ])
        
        # Project each scale to common dimension
        self.scale_projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Random Fourier Features for Hilbert space mapping
        # W ~ N(0, 1/σ²) for Gaussian kernel approximation
        self.register_buffer(
            'rff_weights',
            torch.randn(output_dim, num_rff_features) / sigma
        )
        self.register_buffer(
            'rff_bias',
            torch.rand(num_rff_features) * 2 * np.pi
        )
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(num_rff_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def hilbert_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map features to Hilbert space using Random Fourier Features.
        
        φ(x) = √(2/D) · cos(x·W + b)
        
        This approximates the Gaussian kernel:
        k(x, y) ≈ φ(x)^T · φ(y)
        
        Args:
            x: Input features [B, D]
        
        Returns:
            RFF features [B, num_rff]
        """
        # Linear projection + bias
        projection = torch.matmul(x, self.rff_weights) + self.rff_bias  # [B, num_rff]
        
        # Cosine activation with scaling
        rff = np.sqrt(2.0 / self.num_rff) * torch.cos(projection)
        
        return rff
        
    def forward(self, multi_scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate features across scales.
        
        Args:
            multi_scale_features: List of features [B, N_s, D_s] for each scale
        
        Returns:
            Fused features [B, output_dim]
        """
        assert len(multi_scale_features) == self.num_scales
        
        aggregated_features = []
        attention_weights_list = []
        
        for s, (features, attention, projection) in enumerate(zip(
            multi_scale_features,
            self.scale_attentions,
            self.scale_projections
        )):
            # Attention-weighted aggregation within scale
            scale_feat, attn_weights = attention(features)  # [B, D_s], [B, N_s]
            attention_weights_list.append(attn_weights)
            
            # Project to common dimension
            scale_feat = projection(scale_feat)  # [B, output_dim]
            
            # Map to Hilbert space
            hilbert_feat = self.hilbert_mapping(scale_feat)  # [B, num_rff]
            
            aggregated_features.append(hilbert_feat)
        
        # Weighted combination across scales
        scale_weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * f for w, f in zip(scale_weights, aggregated_features))
        
        # Final projection
        output = self.fusion(fused)  # [B, output_dim]
        
        return output


class PanCANClassifier(nn.Module):
    """
    Complete PanCAN (Panoptic Context Aggregation Network) for image classification.
    
    Architecture Overview:
    ---------------------
    1. Backbone (CvT-W24 or ResNet): Extract multi-scale features
    2. Multi-Order Random Walks: Capture neighborhood context at each scale
    3. Cross-Scale Aggregation: Fuse information across scales
    4. Classification Head: Final prediction
    
    Key Innovations:
    ---------------
    - Hierarchical multi-order context modeling
    - Cross-scale feature aggregation in Hilbert space
    - Attention-based anchor selection
    
    Reference:
        Jiu et al. "Multi-label Classification with Panoptic Context Aggregation Networks"
        arXiv:2512.23486, December 2025
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_walk_orders: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        multi_label: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture ('resnet50', 'resnet101', 'convnext_tiny')
            pretrained: Use pretrained backbone weights
            num_walk_orders: Number of random walk orders (K)
            hidden_dim: Hidden dimension for aggregation
            dropout: Dropout rate
            multi_label: Whether to use multi-label classification
        """
        super().__init__()
        self.num_classes = num_classes
        self.multi_label = multi_label
        
        # Determine out_indices based on backbone type
        # ConvNeXt uses 0-based indices, ResNet uses 1-based
        if 'convnext' in backbone.lower():
            out_indices = (0, 1, 2, 3)
        else:
            out_indices = (1, 2, 3, 4)
        
        # Load backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=out_indices  # 4 scales
        )
        
        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dims = [f.shape[1] for f in features]
        
        print(f"Backbone feature dimensions: {self.feature_dims}")
        
        # Multi-Order Random Walk modules for each scale
        self.random_walks = nn.ModuleList([
            MultiOrderRandomWalk(
                feature_dim=dim,
                num_orders=num_walk_orders,
                dropout=dropout
            ) for dim in self.feature_dims
        ])
        
        # Cross-Scale Aggregation
        self.cross_scale = CrossScaleAggregation(
            feature_dims=self.feature_dims,
            output_dim=hidden_dim,
            num_rff_features=hidden_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classification head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: Whether to return intermediate features
        
        Returns:
            Logits [B, num_classes] (or dict with features if return_features=True)
        """
        # Extract multi-scale features from backbone
        multi_scale_features = self.backbone(x)
        
        # Apply multi-order random walks at each scale
        contextualized_features = []
        for features, rw_module in zip(multi_scale_features, self.random_walks):
            B, C, H, W = features.shape
            
            # Reshape to [B, N, C] where N = H*W
            features_flat = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            
            # Apply random walks
            ctx_features = rw_module(features_flat)  # [B, H*W, C]
            
            contextualized_features.append(ctx_features)
        
        # Cross-scale aggregation
        fused_features = self.cross_scale(contextualized_features)  # [B, hidden_dim]
        
        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        if return_features:
            return {
                'logits': logits,
                'fused_features': fused_features,
                'scale_features': contextualized_features
            }
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Predicted labels [B] (or [B, num_classes] for multi-label)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            
            if self.multi_label:
                predictions = torch.sigmoid(logits) > 0.5
            else:
                predictions = logits.argmax(dim=-1)
        
        return predictions


def create_pancan_model(
    num_classes: int,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    **kwargs
) -> PanCANClassifier:
    """
    Factory function to create PanCAN model.
    
    Args:
        num_classes: Number of output classes
        backbone: Backbone name ('resnet50', 'resnet101', 'convnext_tiny', etc.)
        pretrained: Use pretrained weights
        **kwargs: Additional arguments for PanCANClassifier
    
    Returns:
        PanCANClassifier model
    """
    return PanCANClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        **kwargs
    )


if __name__ == '__main__':
    # Test the model
    print("Testing PanCAN model...")
    
    model = create_pancan_model(num_classes=7, backbone='resnet50')
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with feature return
    output_dict = model(x, return_features=True)
    print(f"\nFeature shapes:")
    print(f"  Logits: {output_dict['logits'].shape}")
    print(f"  Fused: {output_dict['fused_features'].shape}")
    for i, f in enumerate(output_dict['scale_features']):
        print(f"  Scale {i+1}: {f.shape}")
