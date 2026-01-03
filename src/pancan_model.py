"""
PanCAN Model - Complete Implementation

Properly following the paper architecture:
1. Grid-based feature extraction (frozen backbone)
2. Multi-order context aggregation (trainable)
3. Cross-scale aggregation (trainable)
4. Classification head (trainable)

Paper: "Context-aware Aggregation for Image Retrieval"
Key insight: Only context modules are trainable, backbone is FROZEN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from grid_feature_extractor import GridFeatureExtractor, EfficientGridFeatureExtractor
from context_aggregation import MultiOrderContextAggregation
from cross_scale_aggregation import CrossScaleAggregation


class PanCANModel(nn.Module):
    """
    PanCAN: Panoramic Context-Aware Network
    
    Architecture following paper specifications:
    - Backbone: Frozen feature extractor (ResNet101/CvT/TResNet)
    - Grid partition: 8×10, 4×5, 2×3, 1×2, 1×1 (multi-scale)
    - Context aggregation: 2nd order neighbors, 3 layers, threshold 0.71
    - Cross-scale: 2×2 interval aggregation
    
    This is the CORRECT implementation that matches the paper.
    The old implementation incorrectly trained the entire backbone.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'resnet101',
        pretrained: bool = True,
        feature_dim: int = 2048,
        grid_sizes: List[Tuple[int, int]] = None,
        num_orders: int = 2,
        num_layers: int = 3,
        threshold: float = 0.71,
        scale_interval: Tuple[int, int] = (2, 2),
        dropout: float = 0.3,
        efficient_extraction: bool = True
    ):
        """
        Args:
            num_classes: Number of output classes
            backbone_name: timm model name for backbone
            pretrained: Use pretrained weights for backbone
            feature_dim: Feature dimension throughout the model
            grid_sizes: List of grid configurations
            num_orders: Number of random walk orders (paper: 2)
            num_layers: Number of context aggregation layers (paper: 3)
            threshold: Random walk threshold (paper: 0.71)
            scale_interval: Cross-scale aggregation interval (paper: 2×2)
            dropout: Dropout rate for classifier
            efficient_extraction: Use efficient feature extraction
        """
        super().__init__()
        
        # Default grid sizes from paper
        if grid_sizes is None:
            grid_sizes = [(8, 10), (4, 5), (2, 3), (1, 2), (1, 1)]
        
        self.num_classes = num_classes
        self.grid_sizes = grid_sizes
        self.feature_dim = feature_dim
        
        # 1. Grid Feature Extractor (FROZEN backbone)
        ExtractorClass = (EfficientGridFeatureExtractor if efficient_extraction 
                         else GridFeatureExtractor)
        self.feature_extractor = ExtractorClass(
            backbone_name=backbone_name,
            pretrained=pretrained,
            grid_sizes=grid_sizes,
            feature_dim=feature_dim
        )
        
        # 2. Multi-Order Context Aggregation (TRAINABLE)
        # One module per scale
        self.context_modules = nn.ModuleDict({
            f"scale_{h}x{w}": MultiOrderContextAggregation(
                feature_dim=feature_dim,
                num_orders=num_orders,
                num_layers=num_layers,
                threshold=threshold,
                dropout=dropout * 0.5
            )
            for h, w in grid_sizes
        })
        
        # 3. Cross-Scale Aggregation (TRAINABLE)
        self.cross_scale = CrossScaleAggregation(
            feature_dim=feature_dim,
            scale_interval=scale_interval,
            aggregation_type='attention'
        )
        
        # 4. Classification Head (TRAINABLE)
        # Final feature is from coarsest scale (1×1 = single global feature)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Initialize trainable weights
        self._init_weights()
        
        # Print architecture summary
        self._print_summary()
    
    def _init_weights(self):
        """Initialize weights for trainable modules."""
        for module in [self.context_modules, self.cross_scale, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def _print_summary(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print("\n" + "="*60)
        print("PanCAN Model Summary")
        print("="*60)
        print(f"Grid sizes: {self.grid_sizes}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Number of classes: {self.num_classes}")
        print("-"*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters (backbone): {frozen_params:,}")
        print(f"Trainable ratio: {100*trainable_params/total_params:.2f}%")
        print("="*60 + "\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PanCAN.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Class logits (B, num_classes)
        """
        # 1. Extract grid features (backbone frozen)
        grid_output = self.feature_extractor(x)
        multi_scale_features = grid_output['features']
        
        # 2. Apply context aggregation at each scale
        context_features = {}
        for grid_size, features in multi_scale_features.items():
            h, w = grid_size
            module_key = f"scale_{h}x{w}"
            context_features[grid_size] = self.context_modules[module_key](
                features, grid_size
            )
        
        # 3. Cross-scale aggregation
        aggregated = self.cross_scale(context_features)
        
        # 4. Global pooling and classification
        # aggregated should be (B, num_cells, D) for coarsest scale
        global_feature = aggregated.mean(dim=1)  # (B, D)
        
        # 5. Classification
        logits = self.classifier(global_feature)
        
        return logits
    
    def forward_with_features(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning intermediate features for analysis.
        
        Returns:
            Tuple of (logits, feature_dict)
        """
        features_dict = {}
        
        # 1. Extract grid features
        grid_output = self.feature_extractor(x)
        features_dict['grid_features'] = grid_output['all_features']
        
        # 2. Context aggregation
        context_features = {}
        for grid_size, features in grid_output['features'].items():
            h, w = grid_size
            module_key = f"scale_{h}x{w}"
            ctx_feat = self.context_modules[module_key](features, grid_size)
            context_features[grid_size] = ctx_feat
            features_dict[f'context_{h}x{w}'] = ctx_feat
        
        # 3. Cross-scale aggregation
        aggregated = self.cross_scale(context_features)
        features_dict['cross_scale'] = aggregated
        
        # 4. Global pooling
        global_feature = aggregated.mean(dim=1)
        features_dict['global'] = global_feature
        
        # 5. Classification
        logits = self.classifier(global_feature)
        
        return logits, features_dict
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_frozen_params(self) -> int:
        """Return count of frozen parameters (backbone)."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class PanCANLite(nn.Module):
    """
    Lightweight PanCAN for small datasets.
    
    Modifications:
    - Smaller feature dimension
    - Fewer layers
    - Single scale (no multi-scale)
    - More aggressive dropout
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int = 512,
        grid_size: Tuple[int, int] = (4, 5),
        num_orders: int = 2,
        num_layers: int = 2,
        threshold: float = 0.71,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        
        # Simpler feature extractor (single scale)
        self.feature_extractor = EfficientGridFeatureExtractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            grid_sizes=[grid_size],
            feature_dim=feature_dim
        )
        
        # Single context module
        self.context_module = MultiOrderContextAggregation(
            feature_dim=feature_dim,
            num_orders=num_orders,
            num_layers=num_layers,
            threshold=threshold,
            dropout=dropout * 0.5
        )
        
        # Simpler classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Print summary
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[PanCANLite] Trainable params: {trainable:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        grid_output = self.feature_extractor(x)
        features = grid_output['all_features']  # (B, H*W, D)
        
        # Context aggregation
        context_features = self.context_module(features, self.grid_size)
        
        # Global pooling and classification
        global_feature = context_features.mean(dim=1)
        logits = self.classifier(global_feature)
        
        return logits


def create_pancan_model(
    num_classes: int,
    backbone: str = 'resnet101',
    variant: str = 'full',
    **kwargs
) -> nn.Module:
    """
    Factory function to create PanCAN models.
    
    Args:
        num_classes: Number of output classes
        backbone: Backbone model name
        variant: 'full' for PanCANModel, 'lite' for PanCANLite
        **kwargs: Additional arguments passed to model
        
    Returns:
        PanCAN model instance
    """
    if variant == 'full':
        return PanCANModel(
            num_classes=num_classes,
            backbone_name=backbone,
            **kwargs
        )
    elif variant == 'lite':
        return PanCANLite(
            num_classes=num_classes,
            backbone_name=backbone,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
