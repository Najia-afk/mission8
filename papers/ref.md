# Technical Watch - References (January 2026)

## Mission 8: PanCAN and Vision Transformer Papers

**Last Updated:** January 3, 2026  
**Topic:** Multi-label Image Classification, Vision Transformers, CNN vs ViT Comparison

---

## 🔥 Key Papers (June 2025 - January 2026)

### 1. PanCAN: Multi-label Classification with Panoptic Context Aggregation Networks

| Field | Value |
|-------|-------|
| **Authors** | Mingyuan Jiu, Hailong Zhu, Wenchuan Wei, Hichem Sahbi, Rongrong Ji, Mingliang Xu |
| **Date** | December 29, 2025 |
| **arXiv** | [2512.23486](https://arxiv.org/abs/2512.23486) |
| **PDF** | [Download](https://arxiv.org/pdf/2512.23486) |

**Abstract:**
Context modeling is crucial for visual recognition, enabling highly discriminative image representations by integrating both intrinsic and extrinsic relationships between objects and labels in images. This paper introduces the **Deep Panoptic Context Aggregation Network (PanCAN)**, a novel approach that hierarchically integrates multi-order geometric contexts through cross-scale feature aggregation in a high-dimensional Hilbert space.

**Key Contributions:**
- Multi-order neighborhood relationships at each scale using random walks + attention
- Cross-scale feature aggregation with salient anchor selection
- Dynamic fusion via attention mechanism
- State-of-the-art on NUS-WIDE, PASCAL VOC2007, MS-COCO benchmarks

**Relevance to Mission 8:** Direct reference for PanCANLite implementation - context aggregation for multi-label product classification.

---

### 2. Vision Transformers for Image Classification: A Comparative Survey

| Field | Value |
|-------|-------|
| **Authors** | Yaoli Wang, Yaojun Deng, Yuanjin Zheng, Pratik Chattopadhyay, Lipo Wang |
| **Journal** | Technologies 2025, 13(1), 32 |
| **Date** | January 12, 2025 |
| **DOI** | [10.3390/technologies13010032](https://doi.org/10.3390/technologies13010032) |
| **PDF** | [MDPI](https://www.mdpi.com/2227-7080/13/1/32/pdf) |
| **Citations** | 68 |

**Abstract:**
Comprehensive survey on Vision Transformers for image classification. Covers self-attention mechanisms, multi-head attention, ViT architecture, and improved models including:
- CNN concept combinations (CPVT, VT, CvT, CeiT)
- Lightweight designs (LeViT, CCT)
- Deeper structures (CaiT, DeepViT)
- Nested transformers (NesT, CrossViT)

**Key Topics Covered:**
- Self-attention vs Convolution
- Data efficiency improvements (DeiT, T2T-ViT)
- Cross-attention mechanisms
- MetaFormers architecture

**Relevance to Mission 8:** Foundation for understanding ViT-B/16 architecture used in comparison with PanCANLite.

---

### 3. A Deep Ensemble Learning Approach Based on a Vision Transformer and Neural Network for Multi-Label Image Classification

| Field | Value |
|-------|-------|
| **Authors** | Anas W. Abulfaraj, Faisal Binzagr |
| **Journal** | Big Data and Cognitive Computing 2025, 9(2), 39 |
| **Date** | February 11, 2025 |
| **DOI** | [10.3390/bdcc9020039](https://doi.org/10.3390/bdcc9020039) |
| **PDF** | [MDPI](https://www.mdpi.com/2504-2289/9/2/39/pdf) |
| **Citations** | 7 |

**Abstract:**
Proposes a multi-label classification ensemble model combining **Vision Transformer (ViT)** and **CNN** (modified MobileNetV2 and DenseNet201) for detecting multiple objects in images. Uses voting ensemble for final classification.

**Results:**
| Dataset | Accuracy |
|---------|----------|
| PASCAL VOC 2007 | 98.24% |
| PASCAL VOC 2012 | 98.89% |
| MS-COCO | 99.91% |
| NUS-WIDE | 96.69% |

**Key Contributions:**
- ViT for long-range dependencies + local detail
- Modified MobileNetV2 and DenseNet201 with extra conv layers
- Voting ensemble combining all three models

**Relevance to Mission 8:** Directly comparable approach - ensemble of ViT and CNN for multi-label classification like our PanCANLite vs ViT comparison.

---

### 4. Comparative Analysis of Vision Transformers and CNNs for Medical Image Classification

| Field | Value |
|-------|-------|
| **Authors** | Kunal Kawadkar |
| **Date** | July 24, 2025 |
| **arXiv** | [2507.21156](https://arxiv.org/abs/2507.21156) |
| **PDF** | [Download](https://arxiv.org/pdf/2507.21156) |
| **Citations** | 3 |

**Abstract:**
Comprehensive comparison of CNN and ViT architectures across medical imaging tasks. Evaluated ResNet-50, EfficientNet-B0, ViT-Base, and DeiT-Small.

**Key Results:**
| Task | Best Model | Accuracy |
|------|------------|----------|
| Chest X-ray | ResNet-50 | 98.37% |
| Brain Tumor | DeiT-Small | 92.16% |
| Skin Cancer | EfficientNet-B0 | 81.84% |

**Key Finding:** Task-specific model selection is crucial - no single architecture dominates all tasks.

**Relevance to Mission 8:** Validates our approach of comparing CNN (PanCANLite) vs Transformer (ViT-B/16) for specific tasks.

---

## � Multimodal Vision-Language Fusion Papers (2025)

### 5. BERT-ViT-EF & DTCN: Dual Transformer Contrastive Network for Multimodal Analysis

| Field | Value |
|-------|-------|
| **Authors** | Phuong Q. Dao, Mark Roantree, Vuong M. Ngo |
| **Conference** | MEDES 2025 |
| **Date** | October 20, 2025 |
| **arXiv** | [2510.23617](https://arxiv.org/abs/2510.23617) |
| **PDF** | [Download](https://arxiv.org/pdf/2510.23617) |

**Abstract:**
Proposes **BERT-ViT-EF** (Early Fusion) combining BERT for text and ViT for images through early fusion strategy. Extended to **DTCN** (Dual Transformer Contrastive Network) with additional Transformer layer and contrastive learning for cross-modal alignment.

**Results:**
| Dataset | Accuracy | F1-Score |
|---------|----------|----------|
| TumEmo | **78.4%** | **78.3%** |
| MVSA-Single | 76.6% | 75.9% |

**Key Contributions:**
- Early fusion of BERT + ViT encoders
- Contrastive learning for multimodal alignment
- Additional Transformer layer for refined textual context

**Relevance to Mission 8:** Direct architecture inspiration for our ViT + BERT multimodal fusion implementation.

---

### 6. Exploring Fusion Strategies for Multimodal Vision-Language Systems

| Field | Value |
|-------|-------|
| **Authors** | Regan Willis, Jason Bakos |
| **Date** | November 26, 2025 |
| **arXiv** | [2511.21889](https://arxiv.org/abs/2511.21889) |
| **PDF** | [Download](https://arxiv.org/pdf/2511.21889) |

**Abstract:**
Investigates different fusion strategies (early, intermediate, late) using hybrid **BERT + Vision network** frameworks (MobileNetV2 and ViT). Evaluates accuracy vs latency tradeoffs.

**Key Findings:**
| Fusion Strategy | Accuracy | Latency |
|-----------------|----------|---------|
| **Late Fusion** | Highest | Higher |
| **Intermediate** | Medium | Medium |
| **Early Fusion** | Lower | **Lowest** |

**Key Contributions:**
- Systematic comparison of fusion strategies
- BERT + ViT and BERT + MobileNetV2 configurations
- Latency benchmarks on NVIDIA Jetson Orin AGX

**Relevance to Mission 8:** Validates our choice of late fusion strategy for maximum accuracy in product classification.

---

## 📚 Additional Relevant Papers

### 7. EFFResNet-ViT: Fusion-based Explainable Medical Image Classification
- **Source:** IEEE Access 2025
- **DOI:** [10.1109/ACCESS.2025.10938132](https://ieeexplore.ieee.org/abstract/document/10938132/)
- **Focus:** CNN+ViT fusion with explainability (XAI)
- **Citations:** 55

### 6. Explainable AI and Vision Transformers for Brain Tumor Detection
- **Source:** Artificial Intelligence Review 2025
- **DOI:** [10.1007/s10462-025-11221-x](https://link.springer.com/article/10.1007/s10462-025-11221-x)
- **Focus:** Comprehensive XAI + ViT survey
- **Citations:** 7

### 7. Collaborative Low-Rank Adaptation for Pre-Trained Vision Transformers
- **arXiv:** [2512.24603](https://arxiv.org/abs/2512.24603)
- **Date:** December 2025
- **Focus:** ViT fine-tuning methods (relevant to our transfer learning approach)

### 8. Exploring Compositionality in Vision Transformers using Wavelet
- **arXiv:** [2512.24438](https://arxiv.org/abs/2512.24438)
- **Date:** December 2025
- **Focus:** ViT interpretability (relevant to our SHAP analysis)

---

## 🎯 Summary for Mission 8 Technical Watch

| Aspect | PanCANLite (CNN) | ViT-B/16 (Transformer) |
|--------|-----------------|----------------------|
| **Architecture** | Context aggregation, lightweight | Self-attention, patch-based |
| **Parameters** | 3.3M trainable | 526K trainable (frozen backbone) |
| **Our Results** | 84.79% accuracy | 86.69% accuracy |
| **Interpretability** | SHAP + Grad-CAM | SHAP + Saliency maps |
| **Literature Support** | PanCAN paper (arXiv:2512.23486) | ViT surveys (MDPI Technologies) |

### Key Takeaways from Literature:
1. **ViT excels** when combined with transfer learning on small datasets
2. **CNN advantages** remain for lightweight deployment and local feature extraction
3. **Ensemble approaches** (ViT + CNN) often outperform single models
4. **Task-specific selection** is recommended - no universal best architecture
5. **Explainability** (SHAP, attention visualization) is increasingly important

---

## 📎 BibTeX References

```bibtex
@article{jiu2025pancan,
  title={Multi-label Classification with Panoptic Context Aggregation Networks},
  author={Jiu, Mingyuan and Zhu, Hailong and Wei, Wenchuan and Sahbi, Hichem and Ji, Rongrong and Xu, Mingliang},
  journal={arXiv preprint arXiv:2512.23486},
  year={2025}
}

@article{wang2025vit_survey,
  title={Vision Transformers for Image Classification: A Comparative Survey},
  author={Wang, Yaoli and Deng, Yaojun and Zheng, Yuanjin and Chattopadhyay, Pratik and Wang, Lipo},
  journal={Technologies},
  volume={13},
  number={1},
  pages={32},
  year={2025},
  doi={10.3390/technologies13010032}
}

@article{abulfaraj2025ensemble,
  title={A Deep Ensemble Learning Approach Based on a Vision Transformer and Neural Network for Multi-Label Image Classification},
  author={Abulfaraj, Anas W. and Binzagr, Faisal},
  journal={Big Data and Cognitive Computing},
  volume={9},
  number={2},
  pages={39},
  year={2025},
  doi={10.3390/bdcc9020039}
}

@article{kawadkar2025comparative,
  title={Comparative Analysis of Vision Transformers and Convolutional Neural Networks for Medical Image Classification},
  author={Kawadkar, Kunal},
  journal={arXiv preprint arXiv:2507.21156},
  year={2025}
}

@article{dao2025dtcn,
  title={An Enhanced Dual Transformer Contrastive Network for Multimodal Sentiment Analysis},
  author={Dao, Phuong Q. and Roantree, Mark and Ngo, Vuong M.},
  booktitle={MEDES 2025},
  year={2025},
  note={arXiv:2510.23617}
}

@article{willis2025fusion,
  title={Exploring Fusion Strategies for Multimodal Vision-Language Systems},
  author={Willis, Regan and Bakos, Jason},
  journal={arXiv preprint arXiv:2511.21889},
  year={2025}
}
```

---

*Generated for Mission 8 - PanCAN Technical Watch*  
*OpenClassrooms Data Science Path*
