"""
ViT SHAP Analysis with Caching.

Cached SHAP analysis and visualization for Vision Transformer models.
"""

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def load_cached_shap(cache_path):
    """
    Load cached SHAP results.
    
    Args:
        cache_path: Path to the cache file
    
    Returns:
        tuple: (spatial_importance, grid_importance) or (None, None) if not found
    """
    if cache_path.exists():
        print("📦 Loading cached ViT SHAP results...")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print("✅ Loaded from cache!")
        return cache['spatial_importance'], cache['grid_importance']
    return None, None


def save_shap_cache(cache_path, spatial_importance, grid_importance):
    """
    Save SHAP results to cache.
    
    Args:
        cache_path: Path to save cache
        spatial_importance: Spatial importance array
        grid_importance: Grid importance array
    """
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'spatial_importance': spatial_importance,
            'grid_importance': grid_importance
        }, f)
    print("✅ Results cached for future runs!")


def plot_vit_shap_visualization(spatial_importance, grid_importance, save_path=None,
                                 title="Advanced SHAP Global Feature Importance: ViT-B/16"):
    """
    Create enhanced SHAP visualization for ViT model.
    
    Args:
        spatial_importance: 2D array of spatial importance
        grid_importance: 2D array of grid cell importance
        save_path: Optional path to save the figure
        title: Figure title
    
    Returns:
        matplotlib figure
    """
    fig = plt.figure(figsize=(20, 6), facecolor='#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Panel 1: Smoothed spatial importance with jet colormap
    ax1 = plt.subplot(1, 4, 1, facecolor='#16213e')
    spatial_smooth = cv2.GaussianBlur(spatial_importance.astype(np.float32), (11, 11), 0)
    im1 = ax1.imshow(spatial_smooth, cmap='jet')
    ax1.set_title("Global Feature Importance\n(Smoothed SHAP heatmap)", 
                  fontsize=12, color='white', fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
    
    # Panel 2: Attention pattern
    ax2 = plt.subplot(1, 4, 2, facecolor='#16213e')
    overlay_cmap = cm.jet(spatial_smooth)[:, :, :3]
    ax2.imshow(overlay_cmap)
    ax2.set_title("SHAP Attention Pattern\n(Where ViT focuses)", 
                  fontsize=12, color='white', fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Grid importance
    ax3 = plt.subplot(1, 4, 3, facecolor='#16213e')
    im3 = ax3.imshow(grid_importance, cmap='RdYlGn', vmin=0, vmax=grid_importance.max())
    ax3.set_title("Grid-Cell Importance (4×5)\n(Spatial region ranking)", 
                  fontsize=12, color='white', fontweight='bold')
    for i in range(grid_importance.shape[0]):
        for j in range(grid_importance.shape[1]):
            text_color = 'black' if grid_importance[i, j] > 0.5 * grid_importance.max() else 'white'
            ax3.text(j, i, f'{grid_importance[i, j]:.2f}', ha='center', va='center',
                     fontsize=11, color=text_color, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')
    
    # Panel 4: Summary box
    ax4 = plt.subplot(1, 4, 4, facecolor='#16213e')
    ax4.axis('off')
    flat_idx = np.argsort(grid_importance.flatten())[::-1]
    top3_cells = [np.unravel_index(i, grid_importance.shape) for i in flat_idx[:3]]
    
    summary_text = "📊 ViT-B/16 SHAP Summary\n" + "─" * 28 + "\n\n🔝 Top 3 Important Cells:\n"
    for rank, (i, j) in enumerate(top3_cells, 1):
        summary_text += f"   {rank}. Cell ({i},{j}): {grid_importance[i, j]:.3f}\n"
    summary_text += f"\n📈 Statistics:\n   Mean: {grid_importance.mean():.3f}\n   Max: {grid_importance.max():.3f}"
    
    ax4.text(0.05, 0.5, summary_text, fontsize=11, va='center', fontfamily='monospace',
             color='white', bbox=dict(facecolor='#0f3460', alpha=0.8, 
                                      boxstyle='round,pad=0.8', edgecolor='#e94560'))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02, color='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    
    plt.show()
    return fig


def analyze_vit_shap_cached(shap_analyzer, class_names, reports_dir, 
                            plot_global_shap_func=None):
    """
    Complete ViT SHAP analysis with caching support.
    
    Args:
        shap_analyzer: SHAP analyzer instance (or None if using cache)
        class_names: List of class names
        reports_dir: Path to reports directory
        plot_global_shap_func: Function to compute SHAP values (from shap_analysis module)
    
    Returns:
        tuple: (spatial_importance, grid_importance)
    """
    cache_path = reports_dir / 'vit_shap_cache.pkl'
    
    # Try to load from cache
    spatial_importance, grid_importance = load_cached_shap(cache_path)
    
    if spatial_importance is not None:
        # Visualize cached results
        plot_vit_shap_visualization(
            spatial_importance, grid_importance,
            save_path=reports_dir / 'vit_shap_global_importance.png',
            title="Advanced SHAP Global Feature Importance: ViT-B/16 (Cached)"
        )
    else:
        # Compute SHAP values
        print("🔄 Computing ViT SHAP values (first run)...")
        if plot_global_shap_func is None:
            raise ValueError("plot_global_shap_func required for first-time computation")
        
        spatial_importance, grid_importance = plot_global_shap_func(
            analyzer=shap_analyzer,
            class_names=class_names,
            save_dir=reports_dir,
            prefix="vit_"
        )
        
        # Cache results
        save_shap_cache(cache_path, spatial_importance, grid_importance)
    
    return spatial_importance, grid_importance
