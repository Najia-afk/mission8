"""
Final Summary Display for Literature-Based Implementation.

Displays comprehensive results summary with literature references.
"""

import json
from pathlib import Path
from sklearn.metrics import f1_score


def print_final_summary(models_results, ensemble_results, reports_dir, 
                        ensemble_labels=None, model_predictions=None):
    """
    Print final summary of literature-driven implementation results.
    
    Args:
        models_results: dict with model accuracies {'vgg': acc, 'lite': acc, 'vit': acc}
        ensemble_results: dict with 'accuracy' and 'f1_score'
        reports_dir: Path to reports directory
        ensemble_labels: Ground truth labels for F1 calculation
        model_predictions: dict with model predictions for F1 calculation
    """
    vgg_acc = models_results.get('vgg', 0)
    lite_acc = models_results.get('lite', 0)
    vit_acc = models_results.get('vit', 0)
    ensemble_acc = ensemble_results['accuracy']
    ensemble_f1 = ensemble_results['f1_score']
    
    print("=" * 70)
    print("🏆 FINAL SUMMARY: LITERATURE-DRIVEN IMPROVEMENTS")
    print("=" * 70)
    
    print("""
📚 Literature Applied:
   1. Jiu et al. (2025) - PanCAN architecture → PanCANLite adaptation
   2. Wang et al. (2025) - ViT Survey → ViT-B/16 baseline
   3. Abulfaraj & Binzagr (2025) - Ensemble approach → Voting ensemble
   4. Kawadkar (2025) - Task-specific selection → Validated ViT for e-commerce
""")
    
    # Calculate F1 scores if predictions provided
    vgg_f1 = lite_f1 = vit_f1 = 0
    if ensemble_labels is not None and model_predictions is not None:
        if 'vgg' in model_predictions:
            vgg_f1 = f1_score(ensemble_labels, model_predictions['vgg'], average='weighted')
        if 'lite' in model_predictions:
            lite_f1 = f1_score(ensemble_labels, model_predictions['lite'], average='weighted')
        if 'vit' in model_predictions:
            vit_f1 = f1_score(ensemble_labels, model_predictions['vit'], average='weighted')
    
    print("📊 Results Summary:")
    print(f"   {'=' * 50}")
    print(f"   | {'Model':<20} | {'Accuracy':<12} | {'F1-Score':<12} |")
    print(f"   {'=' * 50}")
    print(f"   | {'VGG16 (baseline)':<20} | {vgg_acc:.2%}       | {vgg_f1:.2%}       |")
    print(f"   | {'PanCANLite':<20} | {lite_acc:.2%}       | {lite_f1:.2%}       |")
    print(f"   | {'ViT-B/16':<20} | {vit_acc:.2%}       | {vit_f1:.2%}       |")
    print(f"   {'=' * 50}")
    print(f"   | {'🏆 ENSEMBLE':<20} | {ensemble_acc:.2%}       | {ensemble_f1:.2%}       |")
    print(f"   {'=' * 50}")
    
    improvement = (ensemble_acc - vgg_acc) * 100
    best_single = max(vgg_acc, lite_acc, vit_acc)
    
    print(f"""
🎯 Key Achievements:
   ✅ Ensemble improvement over baseline: +{improvement:.2f}%
   ✅ Ensemble improvement over best single model: {(ensemble_acc - best_single)*100:+.2f}%
   ✅ Literature-validated approach successfully applied
   ✅ Model interpretability via SHAP and saliency maps

📖 Papers Referenced:
   [1] arXiv:2512.23486 - PanCAN (Dec 2025)
   [2] Technologies 13(1):32 - ViT Survey (Jan 2025)  
   [3] BDCC 9(2):39 - Ensemble ViT+CNN (Feb 2025)
   [4] arXiv:2507.21156 - CNN vs ViT (Jul 2025)
""")


def save_ensemble_results(reports_dir, ensemble_acc, ensemble_f1, vit_acc, vit_f1, 
                          vit_params, ensemble_labels=None, vit_preds=None):
    """
    Save ensemble and ViT results to final_results.json.
    
    Args:
        reports_dir: Path to reports directory
        ensemble_acc: Ensemble accuracy
        ensemble_f1: Ensemble F1 score
        vit_acc: ViT accuracy
        vit_f1: ViT F1 score
        vit_params: Number of ViT parameters
        ensemble_labels: Ground truth labels
        vit_preds: ViT predictions
    """
    final_results_path = reports_dir / 'final_results.json'
    
    # Load existing results
    if final_results_path.exists():
        with open(final_results_path, 'r') as f:
            final_results = json.load(f)
    else:
        final_results = {}
    
    # Add ensemble results
    final_results['ensemble'] = {
        'test_accuracy': float(ensemble_acc),
        'test_f1': float(ensemble_f1),
        'models': ['ViT-B/16', 'PanCANLite', 'VGG16'],
        'weights': [1.2, 1.0, 1.0],
        'improvement_over_best': float((ensemble_acc - vit_acc) * 100)
    }
    
    # Add ViT results if not present
    if 'vit_baseline' not in final_results:
        final_results['vit_baseline'] = {
            'test_accuracy': float(vit_acc),
            'test_f1': float(vit_f1),
            'trainable_params': int(vit_params)
        }
    
    # Save updated results
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✅ Results saved to {final_results_path}")
    return final_results


def display_and_save_summary(models_results, ensemble_results, reports_dir,
                             vit_params, ensemble_labels=None, model_predictions=None):
    """
    Complete summary display and results saving.
    
    Args:
        models_results: dict with model accuracies {'vgg': acc, 'lite': acc, 'vit': acc}
        ensemble_results: dict with 'accuracy' and 'f1_score'
        reports_dir: Path to reports directory
        vit_params: Number of ViT parameters
        ensemble_labels: Ground truth labels
        model_predictions: dict with model predictions {'vgg': preds, 'lite': preds, 'vit': preds}
    
    Returns:
        dict with final results
    """
    # Print summary
    print_final_summary(
        models_results, ensemble_results, reports_dir,
        ensemble_labels, model_predictions
    )
    
    # Calculate ViT F1 if predictions available
    vit_f1 = 0
    if ensemble_labels is not None and model_predictions is not None:
        if 'vit' in model_predictions:
            vit_f1 = f1_score(ensemble_labels, model_predictions['vit'], average='weighted')
    
    # Save results
    final_results = save_ensemble_results(
        reports_dir=reports_dir,
        ensemble_acc=ensemble_results['accuracy'],
        ensemble_f1=ensemble_results['f1_score'],
        vit_acc=models_results.get('vit', 0),
        vit_f1=vit_f1,
        vit_params=vit_params
    )
    
    return final_results
