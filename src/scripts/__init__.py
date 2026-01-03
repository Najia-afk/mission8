"""
Plotting and Visualization Scripts for Mission 8
All visualization utilities for the PanCAN project
"""

from .plot_data_exploration import (
    plot_class_distribution,
    plot_sample_images
)

from .plot_training_curves import (
    plot_training_curves_matplotlib,
    plot_training_curves_plotly
)

from .plot_model_comparison import (
    plot_comparison_matplotlib,
    plot_comparison_plotly
)

from .plot_analysis import (
    plot_confusion_matrix,
    analyze_parameter_distribution,
    analyze_param_sample_ratio,
    verify_backbone_freeze,
    print_classification_report,
    print_test_metrics
)

from .interpretability import (
    GradCAMVisualizer,
    SHAPAnalyzer,
    plot_gradcam_grid,
    plot_shap_summary,
    plot_attention_weights
)

from .shap_analysis import (
    SHAPGradientAnalyzer,
    plot_global_shap,
    plot_per_class_shap,
    plot_local_shap,
    print_shap_summary
)

__all__ = [
    # Data exploration
    'plot_class_distribution',
    'plot_sample_images',
    
    # Training curves
    'plot_training_curves_matplotlib',
    'plot_training_curves_plotly',
    
    # Model comparison
    'plot_comparison_matplotlib',
    'plot_comparison_plotly',
    
    # Analysis
    'plot_confusion_matrix',
    'analyze_parameter_distribution',
    'analyze_param_sample_ratio',
    'verify_backbone_freeze',
    'print_classification_report',
    'print_test_metrics',
    
    # Interpretability
    'GradCAMVisualizer',
    'SHAPAnalyzer',
    'plot_gradcam_grid',
    'plot_shap_summary',
    'plot_attention_weights',
    
    # SHAP Analysis (Fast GradientExplainer)
    'SHAPGradientAnalyzer',
    'plot_global_shap',
    'plot_per_class_shap',
    'plot_local_shap',
    'print_shap_summary',
]
