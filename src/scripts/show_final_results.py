"""
MISSION 8 - Complete Results Display Script
Run this to see all model comparison including multimodal fusion.
"""
import json
from pathlib import Path

# Paths
BASE_DIR = Path('/app')
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'

# Load results
results = {}

# Load individual model results if available
final_results_path = REPORTS_DIR / 'final_results.json'
if final_results_path.exists():
    with open(final_results_path, 'r') as f:
        results = json.load(f)

# Load multimodal results
mm_path = MODELS_DIR / 'multimodal_fusion_lite_results.json'
if mm_path.exists():
    with open(mm_path, 'r') as f:
        results['multimodal_fusion'] = json.load(f)

# Display results
print("=" * 70)
print("📊 MISSION 8 - COMPLETE MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<30} {'Accuracy':<12} {'F1 Score':<12}")
print("-" * 70)

models_order = [
    ('pancan_lite', 'PanCANLite (CNN)'),
    ('vgg16_baseline', 'VGG16 (Transfer)'),
    ('vit_b16', 'ViT-B/16'),
    ('ensemble', 'Ensemble (ViT+CNN)'),
    ('multimodal_fusion', '🔗 Multimodal Fusion'),
]

for key, name in models_order:
    if key in results:
        r = results[key]
        acc = r.get('test_accuracy', 0) * 100
        f1 = r.get('test_f1', 0)
        print(f"{name:<30} {acc:.2f}%        {f1:.4f}")

print("=" * 70)

# Best model
if 'multimodal_fusion' in results:
    mm = results['multimodal_fusion']
    print(f"\n🏆 BEST MODEL: Multimodal Fusion")
    print(f"   Accuracy: {mm['test_accuracy']*100:.2f}%")
    print(f"   F1 Score: {mm['test_f1']:.4f}")
    print(f"   Architecture: {mm['architecture']}")
    
    # Calculate improvements
    if 'vit_b16' in results:
        vit_acc = results['vit_b16'].get('test_accuracy', 0)
        improvement = (mm['test_accuracy'] - vit_acc) * 100
        print(f"\n📈 Improvement over ViT-only: +{improvement:.2f}%")

print("\n📚 Scripts reference:")
print("   • src/scripts/vit_baseline.py - Vision Transformer")
print("   • src/scripts/multimodal_fusion_lite.py - Image + Text fusion")
print("   • src/scripts/shap_analysis.py - Model interpretability")
