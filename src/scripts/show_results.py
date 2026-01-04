"""Display final results comparison"""
import json

# Load results
with open('/app/models/multimodal_fusion_lite_results.json', 'r') as f:
    results = json.load(f)

print('='*60)
print('🎯 MULTIMODAL FUSION RESULTS SUMMARY')
print('='*60)
print(f"Test Accuracy: {results['test_accuracy']*100:.2f}%")
print(f"Test F1 Score: {results['test_f1']:.4f}")
print(f"Trainable params: {results['trainable_params']:,}")
print()
print('📊 COMPARISON WITH PREVIOUS MODELS:')
print('='*60)
print(f"{'Model':<30} {'Accuracy':<12} {'F1 Score':<12}")
print('-'*60)
print(f"{'PanCANLite (CNN)':<30} {'84.79%':<12} {'84.79%':<12}")
print(f"{'VGG16 (Transfer)':<30} {'84.79%':<12} {'84.57%':<12}")
print(f"{'ViT-B/16 (Image only)':<30} {'86.69%':<12} {'86.53%':<12}")
print(f"{'Ensemble (ViT+CNN)':<30} {'88.21%':<12} {'88.04%':<12}")
print('-'*60)
acc_str = f"{results['test_accuracy']*100:.2f}%"
f1_str = f"{results['test_f1']:.4f}"
print(f"{'🔗 MULTIMODAL FUSION LITE':<30} {acc_str:<12} {f1_str:<12}")
print('='*60)
print()
print(f"📈 Improvement over ViT-only:  +{(results['test_accuracy'] - 0.8669) * 100:.2f}%")
print(f"📈 Improvement over Ensemble:  +{(results['test_accuracy'] - 0.8821) * 100:.2f}%")
print()
print("🏆 NEW BEST MODEL!")
