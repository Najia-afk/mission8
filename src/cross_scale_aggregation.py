"""
Cross-Scale Aggregation Module for PanCAN

Following paper architecture:
- Aggregate context from micro-level cells to macro-level cells
- Cross-scale interval: 2×2 (Table 4)
- Enables hierarchical context understanding

Paper reference: Section 3.3 - Cross-scale Context Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class CrossScaleAggregation(nn.Module):
    """
    Cross-scale aggregation between different grid scales.
    
    Paper insight: Aggregate context from finer grids (micro) to coarser 
    grids (macro) using 2×2 cell intervals (Table 4).
    
    Example: 8×10 grid cells aggregate to 4×5 grid, then to 2×3, etc.
    """
    
    def __init__(
        self,
        feature_dim: int,
        scale_interval: Tuple[int, int] = (2, 2),
        aggregation_type: str = 'attention'
    ):
        """
        Args:
            feature_dim: Dimension of cell features
            scale_interval: (H_ratio, W_ratio) for scale transition
            aggregation_type: 'attention', 'mean', or 'max'
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.scale_interval = scale_interval
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'attention':
            # Attention-based aggregation
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.ReLU(),
                nn.Linear(feature_dim // 4, 1)
            )
        
        # Feature transformation after aggregation
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        print(f"[CrossScaleAggregation] Interval: {scale_interval}, "
              f"Type: {aggregation_type}")
    
    def aggregate_cells(
        self,
        features: torch.Tensor,
        source_grid: Tuple[int, int],
        target_grid: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Aggregate cells from source grid to target grid.
        
        Args:
            features: Source grid features (B, H_s*W_s, D)
            source_grid: (H_s, W_s) source grid size
            target_grid: (H_t, W_t) target grid size
            
        Returns:
            Aggregated features (B, H_t*W_t, D)
        """
        B = features.shape[0]
        H_s, W_s = source_grid
        H_t, W_t = target_grid
        D = features.shape[-1]
        
        # Reshape to spatial: (B, H_s, W_s, D)
        features_spatial = features.view(B, H_s, W_s, D)
        
        # Calculate cells per target cell
        h_ratio = H_s // H_t
        w_ratio = W_s // W_t
        
        aggregated = []
        
        for i in range(H_t):
            for j in range(W_t):
                # Get source cells that map to this target cell
                h_start = i * h_ratio
                h_end = min((i + 1) * h_ratio, H_s)
                w_start = j * w_ratio
                w_end = min((j + 1) * w_ratio, W_s)
                
                # Extract source cells: (B, h_ratio, w_ratio, D)
                source_cells = features_spatial[:, h_start:h_end, w_start:w_end, :]
                
                # Flatten: (B, num_source_cells, D)
                source_cells = source_cells.reshape(B, -1, D)
                
                # Aggregate
                if self.aggregation_type == 'mean':
                    agg = source_cells.mean(dim=1)
                elif self.aggregation_type == 'max':
                    agg = source_cells.max(dim=1)[0]
                elif self.aggregation_type == 'attention':
                    # Compute attention weights
                    attn_scores = self.attention(source_cells).squeeze(-1)  # (B, num_cells)
                    attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
                    agg = (source_cells * attn_weights).sum(dim=1)
                else:
                    raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
                
                aggregated.append(agg)
        
        # Stack: (B, H_t*W_t, D)
        aggregated = torch.stack(aggregated, dim=1)
        
        return aggregated
    
    def forward(
        self,
        multi_scale_features: Dict[Tuple[int, int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Perform cross-scale aggregation across all scales.
        
        Args:
            multi_scale_features: Dict mapping grid_size to features
            
        Returns:
            Final aggregated feature (B, final_cells, D)
        """
        # Sort scales from finest to coarsest
        scales = sorted(multi_scale_features.keys(), 
                       key=lambda x: x[0] * x[1], reverse=True)
        
        if len(scales) < 2:
            # Only one scale, just return it transformed
            return self.transform(list(multi_scale_features.values())[0])
        
        # Start with finest scale
        current_features = multi_scale_features[scales[0]]
        current_grid = scales[0]
        
        # Progressively aggregate to coarser scales
        for target_grid in scales[1:]:
            # Get features from target scale
            target_features = multi_scale_features[target_grid]
            
            # Aggregate current to target size
            aggregated = self.aggregate_cells(
                current_features, 
                current_grid, 
                target_grid
            )
            
            # Combine with target scale features
            combined = aggregated + target_features
            
            # Transform
            current_features = self.transform(combined)
            current_grid = target_grid
        
        return current_features
    
    def forward_with_skip(
        self,
        multi_scale_features: Dict[Tuple[int, int], torch.Tensor]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Cross-scale aggregation with skip connections.
        
        Returns features at all scales, enhanced with cross-scale context.
        """
        scales = sorted(multi_scale_features.keys(), 
                       key=lambda x: x[0] * x[1], reverse=True)
        
        if len(scales) < 2:
            return multi_scale_features
        
        enhanced_features = {}
        
        # Start with finest scale (no aggregation needed)
        enhanced_features[scales[0]] = self.transform(
            multi_scale_features[scales[0]]
        )
        
        # Propagate context to coarser scales
        for i, target_grid in enumerate(scales[1:], 1):
            # Aggregate from all finer scales
            aggregated_list = []
            
            for source_grid in scales[:i]:
                source_features = enhanced_features[source_grid]
                agg = self.aggregate_cells(
                    source_features,
                    source_grid,
                    target_grid
                )
                aggregated_list.append(agg)
            
            # Combine all aggregated features
            aggregated_combined = torch.stack(aggregated_list, dim=0).mean(dim=0)
            
            # Add target scale features
            combined = aggregated_combined + multi_scale_features[target_grid]
            
            # Transform and store
            enhanced_features[target_grid] = self.transform(combined)
        
        return enhanced_features
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
