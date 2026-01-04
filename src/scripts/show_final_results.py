"""
MISSION 8 - Complete Results Display Script
Display final model comparison with all approaches.
"""
import json
from pathlib import Path


def display_final_comparison(results, reports_dir, save=True):
    """
    Display comprehensive model comparison table and save results.
    
    Args:
        results: dict with model results (pancan_lite, vgg16_baseline, vit_baseline, ensemble, multimodal, dataset)
        reports_dir: Path to save results JSON
        save: Whether to save results to JSON file
    
    Returns:
        dict with best model info
    """
    print("=" * 70)
    print("🏆 FINAL MODEL COMPARISON - ALL APPROACHES")
    print("=" * 70)
    
    # Display comparison table
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Reference'}")
    print("-" * 70)
    
    models_display = [
        ('vgg16_baseline', 'VGG16 (baseline)', 'Baseline'),
        ('pancan_lite', 'PanCANLite', '[Jiu et al., 2025]'),
        ('vit_baseline', 'ViT-B/16', '[Wang et al., 2025]'),
        ('ensemble', 'Ensemble (3-model)', '[Abulfaraj & Binzagr, 2025]'),
        ('multimodal', '🏆 Multimodal Fusion', '[Dao et al., 2025]'),
    ]
    
    best_model = None
    best_acc = 0
    
    for key, name, ref in models_display:
        if key in results:
            r = results[key]
            acc = r.get('test_accuracy', 0)
            f1 = r.get('test_f1', 0)
            print(f"{name:<25} {acc:.2%}       {f1:.2%}       {ref}")
            if acc > best_acc:
                best_acc = acc
                best_model = {'name': name, 'accuracy': acc, 'f1': f1, 'key': key}
    
    print("=" * 70)
    
    # Save results
    if save:
        with open(reports_dir / 'final_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {reports_dir / 'final_results.json'}")
    
    # Print literature references
    print("\n📚 Literature Foundation:")
    print("   [1] Jiu et al., 2025 - PanCAN (arXiv:2512.23486)")
    print("   [2] Wang et al., 2025 - ViT Survey (Technologies 13(1):32)")
    print("   [3] Abulfaraj & Binzagr, 2025 - Ensemble (BDCC 9(2):39)")
    print("   [4] Kawadkar, 2025 - CNN vs ViT (arXiv:2507.21156)")
    print("   [5] Dao et al., 2025 - BERT-ViT-EF (arXiv:2510.23617)")
    print("   [6] Willis & Bakos, 2025 - Fusion Strategies (arXiv:2511.21889)")
    
    if best_model:
        print(f"\n🏆 Best Model: {best_model['name']} ({best_model['accuracy']:.2%})")
    
    return best_model


# Standalone execution
if __name__ == '__main__':
    BASE_DIR = Path('/app')
    MODELS_DIR = BASE_DIR / 'models'
    REPORTS_DIR = BASE_DIR / 'reports'
    
    # Load results from file
    results = {}
    final_results_path = REPORTS_DIR / 'final_results.json'
    if final_results_path.exists():
        with open(final_results_path, 'r') as f:
            results = json.load(f)
    
    # Load multimodal results
    mm_path = MODELS_DIR / 'multimodal_fusion_lite_results.json'
    if mm_path.exists():
        with open(mm_path, 'r') as f:
            results['multimodal'] = json.load(f)
    
    display_final_comparison(results, REPORTS_DIR, save=False)
