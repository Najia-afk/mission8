# Methodological Note: Proof of Concept
## PanCAN (Panoptic Context Aggregation Networks) for E-Commerce Product Classification

---

## 1. Dataset Description (1 page max)

### 1.1 Dataset Presentation

**Name**: Flipkart E-Commerce Product Dataset  
**Source**: Mission 6 - Multimodal Classification Project  
**Period**: 2023-2024

### 1.2 Characteristics

| Characteristic | Value |
|----------------|-------|
| **Number of images** | ~1,050 |
| **Number of categories** | 7 |
| **Original resolution** | Variable |
| **Used resolution** | 224×224 pixels |
| **Format** | RGB (3 channels) |

### 1.3 Class Distribution

| Category | Number of images | Proportion |
|----------|------------------|------------|
| Clothing | XXX | XX% |
| Electronics | XXX | XX% |
| Home & Kitchen | XXX | XX% |
| Footwear | XXX | XX% |
| Accessories | XXX | XX% |
| Sports | XXX | XX% |
| Other | XXX | XX% |

### 1.4 Preprocessing

- **Resizing**: All images resized to 224×224 pixels
- **Normalization**: Values normalized according to ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation**: Rotation (±15°), horizontal flip, zoom (0.9-1.1), color jitter (for training)

---

## 2. Recent Algorithm Concepts (2 pages max)

### 2.1 Introduction to PanCAN

**PanCAN** (Panoptic Context Aggregation Networks) is a recent architecture introduced by Jiu et al. (December 2025) that combines multi-scale CNN features with graph-based context modeling through random walks and cross-scale attention aggregation.

**Reference**: *"Multi-label Classification with Panoptic Context Aggregation Networks"* (arXiv:2512.23486, December 2025)

**Key Innovation**: Unlike traditional CNNs that rely on local receptive fields, PanCAN captures hierarchical neighborhood relationships through multi-order random walks and fuses multi-scale information in a learned Hilbert space using Random Fourier Features.

### 2.2 Detailed Architecture

#### 2.2.1 Multi-Scale Feature Extraction

PanCAN uses a pretrained CNN backbone (ResNet101 or ConvNeXt) to extract features at $S$ scales:

$$\{F_1, F_2, F_3, F_4\} = \text{Backbone}(x)$$

where $F_s \in \mathbb{R}^{H_s \times W_s \times C_s}$ represents features at scale $s$.

For **ResNet101**: $C_s \in \{256, 512, 1024, 2048\}$  
For **ConvNeXt-T**: $C_s \in \{96, 192, 384, 768\}$

#### 2.2.2 Multi-Order Random Walks

Given spatial features $X \in \mathbb{R}^{N \times D}$ where $N = H \cdot W$:

**Step 1 - Affinity Matrix Computation (Gaussian Kernel):**

$$A_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$

where the pairwise distance is computed as:
$$\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2 \cdot x_i \cdot x_j^T$$

**Step 2 - Transition Probability Matrix (Row-Stochastic):**

$$P = D^{-1} \cdot A \quad \text{where } D_{ii} = \sum_j A_{ij}$$

This ensures each row sums to 1, making $P$ a valid probability transition matrix.

**Step 3 - K-hop Context Features:**

$$Z^{(k)} = P^k \cdot X \quad \text{for } k = 1, 2, \ldots, K$$

Each order $k$ captures $k$-hop neighborhood information:
- $k=1$: Immediate neighbors
- $k=2$: 2-hop neighbors
- $k=3$: Extended context

**Step 4 - Order Fusion:**

$$Z_{\text{fused}} = \text{MLP}\left(\text{Concat}\left[W_1 Z^{(1)}, W_2 Z^{(2)}, \ldots, W_K Z^{(K)}\right]\right)$$

where $W_k$ are learnable projection matrices.

#### 2.2.3 Scale Attention Mechanism

For weighting features within each scale:

**Attention Score:**
$$e_i = w^T \cdot \tanh(W \cdot x_i + b)$$

**Softmax Normalization:**
$$\alpha_i = \text{softmax}(e_i) = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$$

**Weighted Aggregation:**
$$z_s = \sum_i \alpha_i \cdot x_i$$

This produces a single representative vector per scale.

#### 2.2.4 Cross-Scale Aggregation via Hilbert Space

PanCAN maps features to a Reproducing Kernel Hilbert Space (RKHS) using **Random Fourier Features** (RFF) for efficient kernel approximation:

**Random Fourier Feature Mapping:**

$$\phi(x) = \sqrt{\frac{2}{D}} \cdot \cos(Wx + b)$$

where:
- $W \sim \mathcal{N}(0, \sigma^{-2}I)$ — random projection weights
- $b \sim \text{Uniform}(0, 2\pi)$ — random bias
- $D$ — number of RFF features

This approximates the Gaussian kernel: $k(x, y) \approx \phi(x)^T \cdot \phi(y)$

**Cross-Scale Fusion:**

$$F_{\text{fused}} = \sum_{s=1}^{S} w_s \cdot \phi_s(F_s) \quad \text{with } w_s = \text{softmax}(\theta_s)$$

where $\theta_s$ are learnable scale importance parameters.

### 2.3 Comparison with CNNs

| Aspect | CNN (VGG16) | PanCAN |
|--------|-------------|--------|
| **Inductive Bias** | Spatial locality, translation equivariance | Graph structure + Attention |
| **Context Modeling** | Local receptive field (progressive) | Multi-order random walks (global) |
| **Multi-Scale** | Single final feature map | Explicit 4-scale aggregation |
| **Complexity** | $O(k^2 \cdot C^2 \cdot HW)$ | $O(N^2 \cdot D)$ for random walks |
| **Parameters** | ~138M (VGG16) | ~45-80M (backbone-dependent) |
| **Transfer Learning** | Effective | Very effective |
| **Interpretability** | Grad-CAM (post-hoc) | Attention weights + Grad-CAM |

### 2.4 Backbone Variants Used

| Backbone | Parameters | Top-1 ImageNet | Notes |
|----------|------------|----------------|-------|
| **ResNet101** | 44.5M | 77.4% | Deep residual learning, 101 layers |
| **ConvNeXt-Tiny** | 28.6M | 82.1% | Modernized ConvNet, Transformer-inspired |

---

## 3. Modeling (2 pages max)

### 3.1 Methodology

#### 3.1.1 Modeling Pipeline

```
Data → Preprocessing → Split → Training → Evaluation → Comparison
```

1. **Preprocessing**:
   - Resize to 224×224
   - Normalization (ImageNet statistics)
   - Data augmentation (train only): RandomHorizontalFlip, RandomRotation(15), ColorJitter

2. **Data Split**:
   - Train: 60%
   - Validation: 15%
   - Test: 25%

3. **Stratification**: Maintaining class distribution across splits

#### 3.1.2 Model Configuration

**VGG16 (Baseline - Mission 6)**:
- Backbone: VGG16 pretrained on ImageNet
- Classification Head: GAP → Dense(1024) → BatchNorm → Dropout(0.5) → Dense(512) → BatchNorm → Dropout(0.3) → Dense(num_classes)
- Optimizer: Adam (lr=1e-4)
- Framework: TensorFlow/Keras

**PanCAN with ResNet101**:
- Backbone: ResNet101 pretrained on ImageNet (features_only=True)
- Feature Scales: 4 (indices 1, 2, 3, 4)
- Feature Dimensions: [256, 512, 1024, 2048]
- Random Walk Orders: K=3
- Hidden Dimension: 512
- RFF Features: 512
- Dropout: 0.3
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Framework: PyTorch + timm

**PanCAN with ConvNeXt-Tiny**:
- Backbone: ConvNeXt-Tiny pretrained on ImageNet
- Feature Scales: 4
- Feature Dimensions: [96, 192, 384, 768]
- Same PanCAN head configuration as above

### 3.2 Evaluation Metric Selection

#### 3.2.1 Main Metrics

| Metric | Formula | Justification |
|--------|---------|---------------|
| **Accuracy** | $\frac{\text{Correct Predictions}}{\text{Total}}$ | Global measure of performance |
| **Macro F1** | $\frac{1}{C}\sum_{c=1}^{C} F1_c$ | Equal weight to each class |
| **Weighted F1** | $\sum_{c=1}^{C} w_c \cdot F1_c$ | Accounts for class imbalance |
| **Cohen's Kappa** | $\kappa = \frac{p_o - p_e}{1 - p_e}$ | Agreement beyond chance |

where $F1_c = 2 \cdot \frac{P_c \cdot R_c}{P_c + R_c}$

#### 3.2.2 Primary Metric Choice

**Macro F1** is selected because:
- It balances precision and recall equally
- It gives equal importance to each class regardless of support
- It is robust to class imbalance in the dataset
- It is standard for multi-class classification benchmarks

### 3.3 Optimization Approach

1. **Baseline**: Training with default hyperparameters from paper
2. **Learning Rate**: Grid search over {5e-5, 1e-4, 3e-4}
3. **Batch Size**: Test with {16, 32} (GPU memory constrained)
4. **Random Walk Orders**: K ∈ {2, 3, 4}
5. **Dropout**: {0.2, 0.3, 0.4}
6. **Weight Decay**: {0.01, 0.05}

### 3.4 Training Configuration

| Parameter | VGG16 (Mission 6) | PanCAN |
|-----------|-------------------|--------|
| **Optimizer** | Adam | AdamW |
| **Learning Rate** | 1e-4 | 1e-4 |
| **Weight Decay** | 0 | 0.01 |
| **Batch Size** | 32 | 16 |
| **Max Epochs** | 20 | 20 |
| **Early Stopping** | patience=5 (val_loss) | patience=5 (val_loss) |
| **LR Scheduler** | ReduceLROnPlateau | CosineAnnealingWarmRestarts |

---

## 4. Results Summary (2 pages max)

### 4.1 Quantitative Results

| Model | Backbone | Accuracy | Macro F1 | Weighted F1 | Params | Training Time |
|-------|----------|----------|----------|-------------|--------|---------------|
| VGG16 (Mission 6) | VGG16 | 0.XX | 0.XX | 0.XX | 138M | XXX s/epoch |
| PanCAN | ResNet101 | 0.XX | 0.XX | 0.XX | ~70M | XXX s/epoch |
| PanCAN | ConvNeXt-T | 0.XX | 0.XX | 0.XX | ~50M | XXX s/epoch |
| **Best Improvement** | - | +X.X% | +X.X% | +X.X% | -XX% | - |

### 4.2 Per-Class Analysis

| Class | F1 (VGG16) | F1 (PanCAN-R101) | F1 (PanCAN-CvT) | Best Δ |
|-------|------------|------------------|-----------------|--------|
| Clothing | 0.XX | 0.XX | 0.XX | +X% |
| Electronics | 0.XX | 0.XX | 0.XX | +X% |
| Home & Kitchen | 0.XX | 0.XX | 0.XX | +X% |
| Footwear | 0.XX | 0.XX | 0.XX | +X% |
| Accessories | 0.XX | 0.XX | 0.XX | +X% |
| Sports | 0.XX | 0.XX | 0.XX | +X% |
| Other | 0.XX | 0.XX | 0.XX | +X% |

### 4.3 Confusion Matrices

*[Insert comparative confusion matrix figures for VGG16, PanCAN-ResNet101, PanCAN-ConvNeXt]*

### 4.4 Qualitative Analysis

**PanCAN Advantages**:
- Multi-order context captures both local and global relationships
- Cross-scale aggregation leverages hierarchical features effectively
- Attention weights provide interpretable scale importance
- Fewer parameters than VGG16 while achieving competitive/better performance

**PanCAN Challenges**:
- Higher memory footprint during training (random walk matrices)
- Longer training time per epoch due to graph computations
- Requires careful tuning of σ (kernel bandwidth) parameter

### 4.5 Conclusion

PanCAN with [ResNet101/ConvNeXt-T] demonstrates an improvement of **+X%** in Macro F1 compared to the VGG16 baseline, confirming the benefit of multi-order random walks and cross-scale aggregation for e-commerce product classification.

---

## 5. Feature Importance - Global and Local (2 pages max)

### 5.1 PanCAN Interpretability Overview

PanCAN provides interpretability through multiple mechanisms:

1. **Scale Attention Weights**: Shows which scales contribute most to predictions
2. **Random Walk Transition Matrices**: Visualizes neighborhood relationships
3. **Grad-CAM**: Compatible with CNN backbone for spatial localization

### 5.2 Global Feature Importance

#### 5.2.1 Scale Importance Analysis

The learned scale weights $w_s = \text{softmax}(\theta_s)$ indicate the relative importance of each feature scale:

| Scale | Resolution | Typical Features | Average Weight |
|-------|------------|------------------|----------------|
| Scale 1 | 56×56 | Fine details, textures | XX% |
| Scale 2 | 28×28 | Parts, local patterns | XX% |
| Scale 3 | 14×14 | Objects, mid-level | XX% |
| Scale 4 | 7×7 | Global, semantic | XX% |

*[Insert scale importance bar chart per class]*

#### 5.2.2 Per-Class Scale Distribution

Different product categories rely on different scales:
- **Clothing**: Higher weight on fine scales (texture patterns)
- **Electronics**: Higher weight on coarse scales (overall shape)
- **Footwear**: Balanced across scales

### 5.3 Local Feature Importance (Individual Examples)

#### 5.3.1 Method: Grad-CAM on PanCAN Backbone

For spatial localization, we apply Grad-CAM to the final backbone feature map:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$$

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c \cdot A^k\right)$$

#### 5.3.2 Attention Weight Visualization

For each image, we visualize the scale attention weights $\alpha_i$ to understand which spatial locations contribute most:

*[Insert 3-4 examples with attention overlay]*

**Example 1: Correctly classified product**
- Attention concentrated on product region
- Background suppressed
- Consistent with human intuition

**Example 2: Misclassified product**
- Scattered attention across image
- Background interference
- Ambiguous product features

### 5.4 Comparison with VGG16 Grad-CAM

| Aspect | PanCAN Interpretability | VGG16 Grad-CAM |
|--------|-------------------------|----------------|
| **Type** | Multi-level (scales + spatial) | Single-level (spatial) |
| **Scale Importance** | Explicit (learned weights) | Not available |
| **Computation** | Forward pass (attention) + Backward (Grad-CAM) | Backward pass only |
| **Resolution** | Multiple (56×56 to 7×7) | Single (7×7 or 14×14) |
| **Context Visibility** | Random walk structure | Not available |

*[Insert side-by-side visual comparison of interpretability maps]*

---

## 6. Limitations and Possible Improvements (1 page max)

### 6.1 Study Limitations

#### 6.1.1 Data-related Limitations

- **Dataset size**: ~1,050 images limits generalization assessment
- **Class imbalance**: Some categories underrepresented
- **Domain gap**: ImageNet pretraining may not be optimal for e-commerce products
- **Image quality**: Variable resolution and lighting conditions

#### 6.1.2 Model-related Limitations

- **Memory constraints**: Random walk matrices scale as $O(N^2)$ with spatial resolution
- **Single hyperparameter sweep**: Limited exploration of σ and K parameters
- **Fixed backbone**: No end-to-end backbone fine-tuning explored

#### 6.1.3 Methodological Limitations

- **POC scope**: Quick proof of concept, not exhaustive benchmark
- **Single seed**: No statistical significance via multiple runs
- **No cross-validation**: Single train/val/test split

### 6.2 Possible Improvements

#### 6.2.1 For Performance

1. **Efficient Random Walks**:
   - Sparse approximation of affinity matrix
   - Nyström method for large-scale graphs

2. **Advanced Backbones**:
   - **EfficientNetV2**: Better efficiency-accuracy trade-off
   - **Swin Transformer**: Hierarchical vision transformer

3. **Data Augmentation**:
   - MixUp, CutMix for regularization
   - RandAugment for automated augmentation

4. **Multimodal Fusion**:
   - Combine image features with product text descriptions
   - CLIP-style contrastive learning

#### 6.2.2 For Interpretability

1. **Attention Flow Analysis**: Track information flow across scales
2. **SHAP Values**: Model-agnostic feature importance
3. **Concept Bottleneck**: Interpretable intermediate concepts
4. **Prototype Networks**: Class-representative examples

#### 6.2.3 For Production Deployment

1. **Model Compression**:
   - Knowledge distillation to smaller backbone
   - Quantization (INT8) for inference speedup

2. **Efficient Inference**:
   - TensorRT / ONNX optimization
   - Batch inference pipeline

3. **Monitoring**:
   - Data drift detection
   - Performance monitoring in production

---

## References

1. Jiu, M., et al. (2025). "Multi-label Classification with Panoptic Context Aggregation Networks." arXiv:2512.23486

2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016. arXiv:1512.03385

3. Liu, Z., et al. (2022). "A ConvNet for the 2020s." CVPR 2022. arXiv:2201.03545

4. Rahimi, A., & Recht, B. (2007). "Random Features for Large-Scale Kernel Machines." NeurIPS 2007.

5. Simonyan, K., & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015. arXiv:1409.1556

6. Selvaraju, R.R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017. arXiv:1610.02391

7. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." ICLR 2019. arXiv:1711.05101

---

*Document prepared as part of Mission 8 - OpenClassrooms Data Scientist Pathway*
*Technique: PanCAN (Panoptic Context Aggregation Networks)*
*Date: December 2025*
