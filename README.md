# Mission 8: Technical Watch - Proof of Concept (POC)
## PanCAN: Panoptic Context Aggregation Networks for Image Classification

[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.23486-b31b1b.svg)](https://arxiv.org/abs/2512.23486)
[![PanCAN](https://img.shields.io/badge/PanCAN-December%202025-purple.svg)](https://arxiv.org/abs/2512.23486)

---

## 📋 Project Context

This project constitutes a **technical watch** on recent modeling techniques in Data Science and AI, carried out as part of Mission 8 of the OpenClassrooms pathway.

The objective is to **test and compare a recent technique** (PanCAN - Panoptic Context Aggregation Networks, published December 2025) with the classical approaches used previously in Mission 6 (VGG16, CNN Transfer Learning).

### Link with Mission 6
This project reuses the **Flipkart dataset** from Mission 6, enabling a direct performance comparison between:
- **Classical approach**: VGG16 Transfer Learning (Mission 6)
- **Recent approach**: PanCAN (Mission 8) - Published December 29, 2025

---

## 🎯 Learning Objectives

1. **Conduct a technical watch on Data Science and AI tools and trends**
   - Consult recognized sources (Arxiv, Papers With Code)
   - Present key points from bibliographic sources
   - Include mathematical details

2. **Implement a Proof of Concept (POC)**
   - Implement the PanCAN architecture
   - Compare with the classical VGG16 approach
   - Analyze results

3. **Write a methodological note**
   - Present the modeling approach
   - Explain evaluation metrics
   - Analyze interpretability (global/local feature importance)
   - Identify limitations and improvements

---

## 📚 State of the Art: PanCAN (Panoptic Context Aggregation Networks)

### Bibliographic References

| Paper | Authors | Year | Citation |
|-------|---------|------|----------|
| **Multi-label Classification with Panoptic Context Aggregation Networks** | Jiu, Zhu, Wei, Sahbi, Ji, Xu | Dec 2025 | [arXiv:2512.23486](https://arxiv.org/abs/2512.23486) |
| **Attention Is All You Need** | Vaswani et al. | 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| **Random Walk Graph Neural Networks** | Nikolentzos et al. | 2020 | [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020) |
| **VGG16: Very Deep Convolutional Networks** | Simonyan & Zisserman | 2014 | [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) |

### PanCAN Principle

```
Image Features (from backbone CNN)
       │
       ▼
┌──────────────────────────────────┐
│  Multi-Scale Feature Extraction  │  → Extract features at different scales
│  (Scale 1, Scale 2, ..., Scale K)│
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Multi-Order Random Walks       │  → Capture neighborhood relationships
│   + Attention Mechanism          │  → Weight important connections
│   (per scale)                    │
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Cross-Scale Aggregation        │  → Select salient anchors
│   (Hilbert Space Mapping)        │  → Fuse features across scales
└──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│   Classification Head            │  → Multi-label / Single-label output
└──────────────────────────────────┘
```

### Key Mathematical Concepts

**Multi-Order Random Walks:**
$$P^{(k)} = P^k \cdot X$$

where $P$ is the transition probability matrix and $k$ is the walk order.

**Attention-Weighted Aggregation:**
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_k \exp(e_{ik})}$$

**Cross-Scale Feature Fusion:**
$$F_{fused} = \sum_{s=1}^{S} w_s \cdot \phi_s(F_s)$$

where $\phi_s$ maps features to a high-dimensional Hilbert space.

---

## 📁 Project Structure

```text
mission8/
├── mission8_pancan_poc.ipynb          # Main POC notebook (PanCAN vs VGG16)
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── docker-compose.yml                  # Docker orchestration
├── Dockerfile                          # Python environment
│
├── dataset/                            # Dataset from mission6
│   └── Flipkart/
│       ├── flipkart_com-ecommerce_sample_1050.csv
│       └── Images/
│
├── src/
│   ├── __init__.py
│   ├── classes/
│   │   ├── __init__.py
│   │   ├── pancan_classifier.py       # PanCAN classifier (ResNet101/ConvNeXt)
│   │   ├── vgg16_baseline.py          # VGG16 baseline (from mission6)
│   │   ├── context_aggregation.py     # Context aggregation + Interpretability
│   │   ├── model_comparison.py        # Comparative analysis utilities
│   │   ├── data_loader.py             # Data loading utilities
│   │   └── metrics_evaluator.py       # Metrics & evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       └── visualization.py           # Plotting utilities
│
├── models/                             # Saved models
│   ├── pancan_resnet101_best.pt       # PanCAN with ResNet101 backbone
│   ├── pancan_convnext_best.pt        # PanCAN with ConvNeXt-Tiny backbone
│   └── vgg16_baseline.keras           # VGG16 baseline model
│
├── reports/                            # Generated reports
│   └── figures/
│
└── note_methodologique/                # Methodological documentation
    └── methodological_note_template.md # PanCAN methodological note
```

---

## 🔬 Approach Comparison

| Aspect | VGG16 (Mission 6) | PanCAN-ResNet101 | PanCAN-ConvNeXt |
|--------|-------------------|------------------|-----------------|
| **Architecture** | CNN (Convolutions) | ResNet101 + Context Agg | ConvNeXt-T + Context Agg |
| **Context Modeling** | Local (receptive field) | Multi-order + Cross-scale | Multi-order + Cross-scale |
| **Key Innovation** | Deep convolutions | Random walks + Attention | Random walks + Attention |
| **Pre-training** | ImageNet | ImageNet (backbone) | ImageNet (backbone) |
| **Parameters** | ~138M | ~70M | ~50M |
| **Interpretability** | Grad-CAM | Attention weights + Grad-CAM | Attention weights + Grad-CAM |
| **Publication** | 2014 | December 2025 | December 2025 |

---

## 🚀 Quick Start

### 1. Installation

```bash
cd mission8

# Create virtual environment
python -m venv mission8_venv
.\mission8_venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

The dataset is already included from mission6:
```
dataset/Flipkart/
├── flipkart_com-ecommerce_sample_1050.csv
└── Images/
```

### 3. Execution

```bash
# Launch Jupyter
jupyter lab mission8_pancan_poc.ipynb
```

### 4. Docker (Alternative)

```bash
docker-compose up --build
```

---

## 📊 Evaluation Metrics

| Metric | Description | Objective |
|--------|-------------|-----------|
| **Accuracy** | Correct classification rate | Maximize |
| **Macro F1** | Average F1-score per class | Evaluate fairness |
| **Weighted F1** | Weighted F1-score | Overall performance |
| **Confusion Matrix** | Error distribution | Identify patterns |
| **Feature Maps** | Visualization of learned features | Interpretability |
| **Attention Weights** | Cross-scale importance | Interpretability |

---

## 📝 Deliverables

1. **POC Notebook** (`mission8_veille_technique.ipynb`)
   - PanCAN implementation
   - Comparison with VGG16
   - Visualizations and analyses

2. **Methodological Note** (PDF, 10 pages max)
   - Dataset description (1 page)
   - Recent algorithm concepts (2 pages)
   - Modeling (2 pages)
   - Results synthesis (2 pages)
   - Global/local feature importance (2 pages)
   - Limitations and improvements (1 page)

3. **Presentation** (30 slides max)
   - Problem reminder
   - Dashboard (mission 7)
   - Technical watch (mission 8)

---

## 🔗 References

- [PanCAN Paper - arXiv:2512.23486](https://arxiv.org/abs/2512.23486)
- [Papers With Code - Image Classification](https://paperswithcode.com/task/image-classification)
- [Random Walk Graph Neural Networks](https://proceedings.neurips.cc/paper/2020)
- [Mission 6 - VGG16 Baseline](../mission6/README.md)

---

*Mission 8 - OpenClassrooms Data Scientist Pathway*
