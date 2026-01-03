"""
Multi-Order Context Aggregation Module for PanCAN

Following paper architecture:
- Build neighborhood graph on grid cells (not pixels!)
- Multi-order random walk: 1st, 2nd order (2nd is optimal per Table 1)
- Threshold for random walk: 0.71 (Table 3)
- Layer depth: 3 layers per scale (Table 2)

Paper reference: Section 3.2 - Context-aware Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class NeighborhoodGraph(nn.Module):
    """
    Build neighborhood graph for grid cells.
    
    For a grid of H×W cells, cells (i,j) and (i',j') are neighbors
    if they are spatially adjacent (4-connectivity or 8-connectivity).
    
    This is DIFFERENT from the old implementation which worked on
    pixel-level CNN features. Here we work on GRID CELLS.
    """
    
    def __init__(self, connectivity: str = '8'):
        """
        Args:
            connectivity: '4' for 4-connectivity, '8' for 8-connectivity
        """
        super().__init__()
        self.connectivity = connectivity
    
    def build_adjacency(
        self, 
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Build adjacency matrix for grid cells.
        
        Args:
            grid_size: (H, W) grid dimensions
            
        Returns:
            Adjacency matrix of shape (H*W, H*W)
        """
        H, W = grid_size
        num_cells = H * W
        
        adj = torch.zeros(num_cells, num_cells)
        
        for i in range(H):
            for j in range(W):
                cell_idx = i * W + j
                
                # Define neighbors based on connectivity
                if self.connectivity == '4':
                    neighbors = [
                        (i-1, j), (i+1, j),  # vertical
                        (i, j-1), (i, j+1)   # horizontal
                    ]
                else:  # 8-connectivity
                    neighbors = [
                        (i-1, j-1), (i-1, j), (i-1, j+1),
                        (i, j-1),             (i, j+1),
                        (i+1, j-1), (i+1, j), (i+1, j+1)
                    ]
                
                for ni, nj in neighbors:
                    if 0 <= ni < H and 0 <= nj < W:
                        neighbor_idx = ni * W + nj
                        adj[cell_idx, neighbor_idx] = 1.0
        
        return adj
    
    def forward(
        self, 
        features: torch.Tensor, 
        grid_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build neighborhood graph from cell features.
        
        Args:
            features: Cell features (B, num_cells, feat_dim)
            grid_size: (H, W) grid dimensions
            
        Returns:
            Tuple of (adjacency_matrix, affinity_matrix)
        """
        device = features.device
        adj = self.build_adjacency(grid_size).to(device)
        
        # Compute affinity (similarity) between neighboring cells
        # Using cosine similarity
        features_norm = F.normalize(features, p=2, dim=-1)
        affinity = torch.bmm(features_norm, features_norm.transpose(1, 2))
        
        # Mask non-neighbors
        adj_expanded = adj.unsqueeze(0).expand(features.shape[0], -1, -1)
        affinity = affinity * adj_expanded
        
        return adj_expanded, affinity


class MultiOrderRandomWalk(nn.Module):
    """
    Multi-order random walk on the cell neighborhood graph.
    
    Paper insight: 2nd order neighbors provide optimal context (Table 1)
    - Order 1: Direct neighbors only
    - Order 2: Neighbors of neighbors (optimal)
    - Order 3+: Adds noise, not recommended
    
    Threshold: 0.71 (Table 3) - controls random walk diffusion
    """
    
    def __init__(
        self,
        num_orders: int = 2,
        threshold: float = 0.71,
        learnable_threshold: bool = False
    ):
        """
        Args:
            num_orders: Number of random walk orders (paper: 2 is optimal)
            threshold: Random walk diffusion threshold (paper: 0.71)
            learnable_threshold: Whether to learn the threshold
        """
        super().__init__()
        self.num_orders = num_orders
        
        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(threshold))
        else:
            self.register_buffer('threshold', torch.tensor(threshold))
    
    def compute_transition_matrix(
        self, 
        affinity: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute random walk transition matrix from affinity.
        
        P[i,j] = affinity[i,j] / sum_k(affinity[i,k])
        
        Args:
            affinity: Affinity matrix (B, N, N)
            adjacency: Binary adjacency matrix (B, N, N)
            
        Returns:
            Transition matrix (B, N, N)
        """
        # Apply threshold to affinity
        affinity_thresholded = affinity * (affinity > self.threshold).float()
        
        # Mask by adjacency
        affinity_masked = affinity_thresholded * adjacency
        
        # Row-normalize to get transition probabilities
        row_sums = affinity_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        transition = affinity_masked / row_sums
        
        return transition
    
    def forward(
        self,
        features: torch.Tensor,
        affinity: torch.Tensor,
        adjacency: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Perform multi-order random walk aggregation.
        
        Args:
            features: Cell features (B, N, D)
            affinity: Affinity matrix (B, N, N)
            adjacency: Adjacency matrix (B, N, N)
            
        Returns:
            List of aggregated features for each order
        """
        B, N, D = features.shape
        
        # Compute transition matrix
        P = self.compute_transition_matrix(affinity, adjacency)
        
        # Multi-order random walk
        aggregated_features = [features]  # Order 0 = original
        
        P_power = P.clone()
        for order in range(1, self.num_orders + 1):
            # Random walk: X' = P @ X
            agg_feat = torch.bmm(P_power, features)
            aggregated_features.append(agg_feat)
            
            # Higher order: P^k
            if order < self.num_orders:
                P_power = torch.bmm(P_power, P)
        
        return aggregated_features


class ContextAggregationLayer(nn.Module):
    """
    Single layer of context-aware aggregation.
    
    Combines multi-order random walk features with learned attention.
    Paper uses 3 layers per scale (Table 2).
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_orders: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of cell features
            num_orders: Number of random walk orders
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_orders = num_orders
        
        # Attention weights for combining orders
        self.order_attention = nn.Sequential(
            nn.Linear(feature_dim * (num_orders + 1), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_orders + 1),
            nn.Softmax(dim=-1)
        )
        
        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        features: torch.Tensor,
        multi_order_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregate multi-order features with attention.
        
        Args:
            features: Original cell features (B, N, D)
            multi_order_features: List of order-k aggregated features
            
        Returns:
            Context-enhanced features (B, N, D)
        """
        B, N, D = features.shape
        
        # Stack multi-order features: (B, N, num_orders+1, D)
        stacked = torch.stack(multi_order_features, dim=2)
        
        # Compute attention weights
        concat = stacked.reshape(B, N, -1)  # (B, N, (num_orders+1)*D)
        attn_weights = self.order_attention(concat)  # (B, N, num_orders+1)
        
        # Weighted combination
        attn_weights = attn_weights.unsqueeze(-1)  # (B, N, num_orders+1, 1)
        aggregated = (stacked * attn_weights).sum(dim=2)  # (B, N, D)
        
        # Transform
        transformed = self.transform(aggregated)
        
        # Residual connection
        output = self.residual_weight * transformed + (1 - self.residual_weight) * features
        
        return output


class MultiOrderContextAggregation(nn.Module):
    """
    Full multi-order context aggregation module.
    
    Paper architecture:
    - Order: 2 (2nd order neighbors, Table 1)
    - Depth: 3 layers (Table 2)
    - Threshold: 0.71 (Table 3)
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_orders: int = 2,
        num_layers: int = 3,
        threshold: float = 0.71,
        dropout: float = 0.1,
        connectivity: str = '8'
    ):
        """
        Args:
            feature_dim: Dimension of cell features
            num_orders: Number of random walk orders (paper: 2)
            num_layers: Number of aggregation layers (paper: 3)
            threshold: Random walk threshold (paper: 0.71)
            dropout: Dropout probability
            connectivity: Graph connectivity ('4' or '8')
        """
        super().__init__()
        
        self.num_orders = num_orders
        self.num_layers = num_layers
        
        # Neighborhood graph builder
        self.graph_builder = NeighborhoodGraph(connectivity)
        
        # Multi-order random walk
        self.random_walk = MultiOrderRandomWalk(num_orders, threshold)
        
        # Aggregation layers
        self.layers = nn.ModuleList([
            ContextAggregationLayer(feature_dim, num_orders, dropout)
            for _ in range(num_layers)
        ])
        
        print(f"[MultiOrderContextAggregation] Orders: {num_orders}, "
              f"Layers: {num_layers}, Threshold: {threshold}")
    
    def forward(
        self,
        features: torch.Tensor,
        grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Apply multi-order context aggregation.
        
        Args:
            features: Cell features (B, num_cells, feat_dim)
            grid_size: (H, W) grid dimensions
            
        Returns:
            Context-enhanced features (B, num_cells, feat_dim)
        """
        # Build neighborhood graph
        adjacency, affinity = self.graph_builder(features, grid_size)
        
        # Apply aggregation layers
        x = features
        for layer in self.layers:
            # Compute multi-order random walk features
            multi_order_feats = self.random_walk(x, affinity, adjacency)
            
            # Aggregate with attention
            x = layer(x, multi_order_feats)
            
            # Update affinity for next layer
            x_norm = F.normalize(x, p=2, dim=-1)
            affinity = torch.bmm(x_norm, x_norm.transpose(1, 2))
            affinity = affinity * adjacency
        
        return x
    
    def get_trainable_params(self) -> int:
        """Return count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
