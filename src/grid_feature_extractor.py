"""
Grid-based Feature Extractor for PanCAN

Following the paper architecture:
- Partition images into grid cells (e.g., 8×10, 4×5, 2×3, 1×2, 1×1)
- Extract frozen CNN features from each cell
- This is the CORRECT approach: backbone is FROZEN, only context modules train

Paper reference: Section 3.1 - Multi-scale Feature Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple, Dict, Optional
import numpy as np


class GridFeatureExtractor(nn.Module):
    """
    Extract features from grid cells of images using a frozen CNN backbone.
    
    Key insight from paper: The backbone (ResNet101, CvT, etc.) is used ONLY
    for feature extraction. It is NOT trained. Only the context-aware modules
    are trainable.
    
    Grid configurations follow paper's optimal settings:
    - Multi-scale grids: 8×10, 4×5, 2×3, 1×2, 1×1
    - Features extracted at each scale independently
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet101',
        pretrained: bool = True,
        grid_sizes: List[Tuple[int, int]] = None,
        feature_dim: int = 2048,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            backbone_name: Name of the backbone model (timm format)
            pretrained: Use ImageNet pretrained weights
            grid_sizes: List of (H, W) grid configurations
            feature_dim: Output feature dimension per cell
            input_size: Expected input image size
        """
        super().__init__()
        
        # Default grid sizes from paper (Table 4)
        if grid_sizes is None:
            grid_sizes = [(8, 10), (4, 5), (2, 3), (1, 2), (1, 1)]
        
        self.grid_sizes = grid_sizes
        self.feature_dim = feature_dim
        self.input_size = input_size
        
        # Create frozen backbone
        self.backbone = self._create_backbone(backbone_name, pretrained)
        
        # CRITICAL: Freeze backbone - this is the paper's approach
        self._freeze_backbone()
        
        # Get actual feature dimension from backbone
        self._feature_dim_backbone = self._get_backbone_feature_dim()
        
        # Projection to common feature dimension if needed
        if self._feature_dim_backbone != feature_dim:
            self.projection = nn.Linear(self._feature_dim_backbone, feature_dim)
        else:
            self.projection = nn.Identity()
        
        # Store grid cell counts for each scale
        self.grid_cell_counts = [h * w for h, w in grid_sizes]
        self.total_cells = sum(self.grid_cell_counts)
        
        print(f"[GridFeatureExtractor] Backbone: {backbone_name}")
        print(f"[GridFeatureExtractor] Backbone frozen: True")
        print(f"[GridFeatureExtractor] Grid sizes: {grid_sizes}")
        print(f"[GridFeatureExtractor] Total grid cells: {self.total_cells}")
        print(f"[GridFeatureExtractor] Feature dim: {feature_dim}")
    
    def _create_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Create backbone model using timm."""
        model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Remove global pooling
        )
        return model
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters - CRITICAL for correct implementation."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("[GridFeatureExtractor] Backbone frozen - no gradient updates")
    
    def _get_backbone_feature_dim(self) -> int:
        """Determine feature dimension from backbone."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *self.input_size)
            features = self.backbone(dummy)
            return features.shape[1]
    
    def extract_cell_features(
        self, 
        x: torch.Tensor, 
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Extract features for each cell in a grid.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            grid_size: (num_rows, num_cols) for grid partition
            
        Returns:
            Cell features of shape (B, num_cells, feature_dim)
        """
        B, C, H, W = x.shape
        gh, gw = grid_size
        
        # Calculate cell size
        cell_h = H // gh
        cell_w = W // gw
        
        cell_features = []
        
        for i in range(gh):
            for j in range(gw):
                # Extract cell region
                h_start = i * cell_h
                h_end = (i + 1) * cell_h if i < gh - 1 else H
                w_start = j * cell_w
                w_end = (j + 1) * cell_w if j < gw - 1 else W
                
                cell = x[:, :, h_start:h_end, w_start:w_end]
                
                # Resize to backbone expected size if needed
                if cell.shape[2:] != self.input_size:
                    cell = F.interpolate(
                        cell, 
                        size=self.input_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # Extract features (backbone is frozen)
                # Use set_grad_enabled to allow Grad-CAM when x requires grad
                with torch.set_grad_enabled(x.requires_grad):
                    feat = self.backbone(cell)
                    
                # Global average pool if spatial dimensions exist
                if feat.dim() == 4:
                    feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                
                cell_features.append(feat)
        
        # Stack: (B, num_cells, feat_dim)
        cell_features = torch.stack(cell_features, dim=1)
        
        # Project to common dimension
        cell_features = self.projection(cell_features)
        
        return cell_features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale grid features.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Dictionary with:
                - 'features': Dict mapping grid_size to cell features
                - 'all_features': Concatenated features (B, total_cells, feat_dim)
        """
        # Ensure backbone stays in eval mode
        self.backbone.eval()
        
        features_dict = {}
        all_features = []
        
        for grid_size in self.grid_sizes:
            cell_feats = self.extract_cell_features(x, grid_size)
            features_dict[grid_size] = cell_feats
            all_features.append(cell_feats)
        
        # Concatenate all scales
        all_features = torch.cat(all_features, dim=1)
        
        return {
            'features': features_dict,
            'all_features': all_features,
            'grid_sizes': self.grid_sizes,
            'cell_counts': self.grid_cell_counts
        }
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters (should be minimal)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return total parameters including frozen."""
        return sum(p.numel() for p in self.parameters())


class EfficientGridFeatureExtractor(GridFeatureExtractor):
    """
    Memory-efficient version that processes images once and partitions
    the feature map instead of extracting features from each cell separately.
    
    More efficient for inference but requires the backbone to produce
    spatially-aligned feature maps.
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet101',
        pretrained: bool = True,
        grid_sizes: List[Tuple[int, int]] = None,
        feature_dim: int = 2048,
        input_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__(
            backbone_name, pretrained, grid_sizes, feature_dim, input_size
        )
        
        # Get spatial output size of backbone
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)
            feat_map = self.backbone(dummy)
            if feat_map.dim() == 4:
                self.feat_spatial_size = feat_map.shape[2:]
            else:
                self.feat_spatial_size = None
                print("[Warning] Backbone doesn't produce spatial features, "
                      "falling back to cell-by-cell extraction")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale grid features efficiently.
        
        Instead of running backbone on each cell, we:
        1. Run backbone once on full image
        2. Partition the feature map into grid cells
        3. Pool each cell to get cell features
        """
        # Ensure backbone stays in eval mode
        self.backbone.eval()
        
        # If no spatial features, fall back to parent implementation
        if self.feat_spatial_size is None:
            return super().forward(x)
        
        # Extract full feature map once
        # Use set_grad_enabled to allow gradients when x requires grad (for interpretability)
        # while keeping backbone parameters frozen during training
        with torch.set_grad_enabled(x.requires_grad):
            feat_map = self.backbone(x)  # (B, C, H', W')
        
        B, C, fH, fW = feat_map.shape
        
        features_dict = {}
        all_features = []
        
        for grid_size in self.grid_sizes:
            gh, gw = grid_size
            
            # Calculate cell size in feature map
            cell_h = fH // gh
            cell_w = fW // gw
            
            cell_features = []
            
            for i in range(gh):
                for j in range(gw):
                    # Extract cell from feature map
                    h_start = i * cell_h
                    h_end = (i + 1) * cell_h if i < gh - 1 else fH
                    w_start = j * cell_w
                    w_end = (j + 1) * cell_w if j < gw - 1 else fW
                    
                    cell = feat_map[:, :, h_start:h_end, w_start:w_end]
                    
                    # Global average pool the cell
                    cell_feat = F.adaptive_avg_pool2d(cell, 1).flatten(1)
                    cell_features.append(cell_feat)
            
            # Stack: (B, num_cells, feat_dim)
            cell_features = torch.stack(cell_features, dim=1)
            cell_features = self.projection(cell_features)
            
            features_dict[grid_size] = cell_features
            all_features.append(cell_features)
        
        # Concatenate all scales
        all_features = torch.cat(all_features, dim=1)
        
        return {
            'features': features_dict,
            'all_features': all_features,
            'grid_sizes': self.grid_sizes,
            'cell_counts': self.grid_cell_counts
        }
