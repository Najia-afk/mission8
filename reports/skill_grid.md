
# Mission 8 - Skills Assessment Grid

## Project overview

**Title**: Technology Watch – ViT, Ensemble, and Multimodal Fusion for E-commerce Image Classification (with PanCANLite study)  
**Dataset**: Flipkart E-commerce (1050 images, 7 product categories)  
**Techniques studied**: PanCAN / PanCANLite, Vision Transformer (ViT-B/16), voting ensemble, multimodal fusion (image + text)  
**Main references (2025)**: Jiu et al. (PanCAN), Wang et al. (ViT survey), Kawadkar (CNN vs ViT), Abulfaraj & Binzagr (ensemble), Dao et al. + Willis & Bakos (fusion)  
**Date**: January 2026  

---

## Competency 1: Conduct a technology watch on tools and trends in data science and AI

### CE1: Reputable, recently produced sources ✅ VALIDATED

| Criterion | Status | Rationale |
|---------|------|---------------|
| **Main sources** | ✅ Excellent | 6 research references (2025), including peer-reviewed journal papers (MDPI) and widely-used preprints (arXiv) |
| **Journal/Conference** | ✅ Validated | Mix of journal articles and arXiv preprints (explicitly listed in the notebook) |
| **Topical relevance** | ✅ Excellent | PanCAN + ViT + ensemble + multimodal fusion, directly aligned with e-commerce classification constraints |
| **Citations** | ⚠️ To verify | 2025 papers: citations still emerging for some preprints |

**Evidence in the notebook**:
- Cell 1 (markdown): “Literature Foundation” lists 6 papers and the full bibliography is included in the References section
- Sections 5–13: Concepts and implementations tied back to the papers (PanCAN / ViT / ensemble / fusion)

**Recommendation**: ✅ Criterion satisfied – recent and relevant academic source

---

### CE2: Key points presented, including mathematical details ✅ VALIDATED

| Aspect | Status | Location |
|--------|------|--------------|
| **Core concepts** | ✅ Excellent | Section 5: Understanding PanCAN Architecture |
| **Mathematical details** | ✅ Good | Explanation of neighborhood orders, hierarchical grids |
| **Technical architecture** | ✅ Excellent | Frozen backbone, feature dimensions, grid sizes |
| **Optimal parameters** | ✅ Excellent | Table with threshold, num_orders, num_layers (from the paper) |

**Evidence in the notebook**:

```markdown
Section 5.1: What is PanCAN?
- Multi-Order Context: 1st order (neighbors), 2nd order (extended range)
- Cross-Scale Aggregation: 8×10 → 4×5 → 2×3 → 1×2 → 1×1
- Random Walk + Attention Mechanism

Section 5.3: Architecture Variants
- Feature dimensions: 2048 (full) vs 512 (lite)
- Grid scales: 5 scales (full) vs 1 scale (lite)
- Parameters: 108-260M (full) vs 3-5M (lite)
```

**Mathematical details included**:
- Parameter-to-sample ratio: 172,700:1 (full) vs 5,226:1 (lite)
- Numerical stability analysis (NaN losses)
- Feature dimensionality and its impact on convergence
- Multi-order neighborhoods: 1st- and 2nd-order neighborhood graphs

**Recommendation**: ✅ Criterion satisfied – good mathematical depth with clear explanations

---

### CE3: Proof of concept (PoC) with comparison ✅ VALIDATED

| Item | Status | Description |
|---------|------|-------------|
| **PoC implemented** | ✅ Excellent | Multiple working baselines + literature-driven extensions (ViT, ensemble, multimodal fusion) |
| **Classic baseline** | ✅ Excellent | VGG16 with transfer learning |
| **Transformer baseline** | ✅ Excellent | ViT-B/16 baseline |
| **Comparison metrics** | ✅ Excellent | Accuracy, F1-score, parameter efficiency, training status |
| **Tests on real data** | ✅ Excellent | Flipkart dataset (629 train, 158 val, 263 test) |

**PoC results (final summary in the notebook)**:

| Model | Test Accuracy | F1-score | Training Status |
|--------|--------------|----------|-----------------|
| VGG16 (baseline) | 84.79% | 84.66% | ✅ Converged |
| PanCANLite | 84.79% | 84.68% | ✅ Converged |
| ViT-B/16 | 86.69% | 86.54% | ✅ Converged |
| Ensemble (ViT + PanCANLite + VGG16) | 88.21% | 87.95% | ✅ Evaluated |
| **Multimodal Fusion (image + text)** | **92.40%** | **92.15%** | ✅ Evaluated |
| Full PanCAN | N/A | N/A | ❌ NaN losses |

**Key takeaway**: the study ends with ViT-based methods (ensemble and multimodal fusion) as the most competitive options on this dataset, while full PanCAN is not feasible at this scale.

**Implemented code**:
- `src/grid_feature_extractor.py`: grid-based feature extraction
- `src/context_aggregation.py`: multi-order aggregation
- `src/cross_scale_aggregation.py`: cross-scale fusion
- `src/pancan_model.py`: full PanCAN and PanCANLite models
- `src/data_loader.py`: loading and preprocessing
- `src/trainer.py`: training pipeline
- `src/scripts/vit_baseline.py`: ViT baseline training/evaluation and comparison plots
- `src/scripts/saliency_visualization.py`: saliency-map visualizations (CNN/ViT)
- `src/scripts/shap_analysis.py`: SHAP gradient-based analyzer + plotting helpers
- `src/scripts/vit_shap_cached.py`: cached ViT SHAP workflows
- `src/ensemble.py`: soft voting ensemble (literature-based)
- `src/scripts/multimodal_fusion_lite.py`: lightweight multimodal training entrypoint
- `src/scripts/multimodal_evaluation.py`: multimodal evaluation + report

**Recommendation**: ✅ Excellent – complete PoC with production-ready code

---

## Competency 2: Write a methodological note

### CE1: Concise modeling approach in a methodological note ⚠️ PARTIALLY VALIDATED

| Criterion | Status | Rationale |
|---------|------|---------------|
| **Formal methodological note** | ⚠️ Missing | No separate PDF document following the provided template |
| **Documentation in the notebook** | ✅ Excellent | Very complete markdown sections (Sections 5–9) |
| **Summary of the approach** | ✅ Good | Clearly covers preprocessing, architecture, training |

**Content already present in the notebook** (to convert into a formal note):

1. **Dataset** (Section 4):
   - 7 e-commerce categories
   - 1050 images (629/158/263 split)
   - Data augmentation: rotation, flip, color jitter

2. **Methodology** (Section 5):
   - Frozen ResNet50 backbone
   - Grid-based feature extraction
   - Multi-order context aggregation
   - Comparison full PanCAN vs PanCANLite

3. **Results** (Sections 6–7):
   - Detailed comparison with tables and charts
   - Failure analysis (NaN losses)

**Recommendation**: ⚠️ Create a formal PDF document `note_methodologique.pdf` based on the notebook sections

---

### CE2: Evaluation metrics and optimization ✅ VALIDATED

| Aspect | Status | Details |
|--------|------|---------|
| **Evaluation metrics** | ✅ Excellent | Accuracy, macro F1-score, Precision, Recall |
| **Metric rationale** | ✅ Good | Macro F1-score for balanced classes |
| **Optimization approach** | ✅ Excellent | Learning rate scheduling, early stopping, dropout |
| **Hyperparameters** | ✅ Excellent | Grid size, feature dim, num_layers, threshold |

**Evidence in the notebook**:

```python
# Optimal configuration (configuration cell in the notebook)
CONFIG = {
    'learning_rate': 1e-4,      # Reduced for numerical stability
    'weight_decay': 1e-4,
    'num_epochs': 30,
    'patience': 10,              # Early stopping
    'dropout': 0.5,              # PanCANLite - strong regularization
    'label_smoothing': 0.1,
    'gradient_clip': 1.0,
}

# Metrics tracking (trainer.py)
- Train/Val Loss
- Train/Val Accuracy
- Learning rate schedule (ReduceLROnPlateau)
- Best model checkpoint saving
```

**Documented optimization process**:
1. Attempt with full PanCAN → failure (NaN losses)
2. Reduced complexity: 5 scales → 3 scales
3. Switched to PanCANLite: 1 scale, 512 features
4. Result: stable convergence and improved performance

**Recommendation**: ✅ Excellent – well documented iterative process

---

### CE3: Global and local interpretability ❌ NOT ADDRESSED

### CE3: Global and local interpretability ✅ VALIDATED

| Aspect | Status | Evidence |
|--------|------|----------|
| **Global feature importance** | ✅ Good | ViT SHAP analysis (global + per-class), cached figures in `reports/` |
| **Local feature importance** | ✅ Good | ViT local SHAP explanations (with caching) |
| **Attribution visualization** | ✅ Excellent | ViT saliency maps exported to `reports/vit_saliency_maps.png` |
| **Error analysis** | ✅ Good | Confusion matrix analysis via refactored script + confidence pattern analysis |

**Evidence in the notebook**:
- ViT saliency maps: `src/scripts/saliency_visualization.py`
- ViT SHAP pipeline: `src/scripts/shap_analysis.py` + `src/scripts/vit_shap_cached.py`
- Confusion matrix + confidence patterns: `analyze_confusion_matrix(...)` + `src/scripts/confidence_analysis.py`

**Recommendation**: ✅ Criterion satisfied – include the generated figures in the final methodological note PDF

---

### CE4: Limitations and potential improvements ✅ VALIDATED

| Aspect | Status | Location |
|--------|------|--------------|
| **Limitations identified** | ✅ Excellent | Section 9.1: Limitations Discovered |
| **Dataset constraints** | ✅ Excellent | Analysis: 629 samples vs 80K+ required |
| **Failures documented** | ✅ Excellent | Full PanCAN failure analysis |
| **Improvements proposed** | ✅ Excellent | Section 9.6: Future Directions |

**Documented limitations**:

1. **Dataset scale**:
   - 629 samples is insufficient for full PanCAN (needs >50K)
   - Critical parameter/sample ratio (172,700:1)
   - Numerical instability (NaN losses)

2. **Architecture**:
   - Multi-scale hierarchies require statistical diversity
   - High feature dimensionality is problematic
   - 3rd-order neighborhoods become too sparse

3. **Performance**:
   - Gap vs Mission 6 multi-modal reference (95.04% vs 92.40%)
   - Lightweight text encoding (TF‑IDF) may cap performance vs richer encoders

**Proposed improvements** (Section 9.6):

```markdown
1. Hybrid approach: PanCANLite + text features → target 95%+
2. Data augmentation: MixUp, CutMix
3. Semi-supervised learning: unlabeled product images
4. Efficient architectures: MobileNet-based PanCANLite
5. Production optimization: quantization, pruning
```

**Recommendation**: ✅ Excellent – mature critical analysis with concrete proposals

---

## Competency 3: Oral presentation of a modeling approach

### CE1: Explanation understandable for a non-technical audience ✅ VALIDATED

| Aspect | Status | Rationale |
|--------|------|---------------|
| **Concept popularization** | ✅ Excellent | Clear metaphors and analogies |
| **Evaluation method** | ✅ Excellent | Accuracy/F1 explained simply |
| **Interpretation of results** | ✅ Good | Clear comparison across vision-only, ensemble, and multimodal |
| **Feature importance** | ✅ Good | Saliency + SHAP figures produced and reusable in slides |

**Popularization elements included**:

```markdown
Section 5.1: "What is PanCAN?"
- Analogy: "First-order = direct neighbors, Second-order = neighbors of neighbors"
- Visualization: Hierarchical grids 8×10 → 4×5 → 2×3 → 1×2 → 1×1
- Business context: "Captures relationships between product features at different scales"

Section 6.2: Key Findings
- ✅ Clear “best model” conclusion (multimodal fusion)
- Quantified progression: vision-only (VGG/PanCANLite/ViT) → ensemble → multimodal
- Efficiency insight: "97% fewer parameters" (PanCANLite vs VGG)
- 4-panel comparison visualization (log scales, ratios, threshold lines)
```

**Accessible language**:
- ✅ "Micro-contexts (fine details) → Macro-contexts (global structures)"
- ✅ "Parameter/sample ratio becomes critical"
- ✅ "Model complexity must scale with dataset size"

**Recommendation**: ✅ Criterion satisfied – good balance between accessibility and technical precision

---

### CE2: Simple answers to questions ✅ VALIDATED (prepared)

| Question type | Preparation | Location |
|------------------|-------------|--------------|
| **"Why these models?"** | ✅ Prepared | Sections 5–13: PanCANLite, ViT, ensemble, multimodal |
| **"Why did it fail?"** | ✅ Prepared | Section 6.2 + 8.3: Full PanCAN failure analysis |
| **"How to improve?"** | ✅ Prepared | Section 9.6: Future Directions (5 items) |
| **"What’s different vs Mission 6?"** | ✅ Prepared | Section 7: Comparison with Mission 6 |

**Prepared example answers**:

**Q: "Why evaluate ViT on this task?"**
> "ViT models global context via self-attention. On our Flipkart dataset, ViT-B/16 outperforms the CNN baselines (86.69% vs 84.79%), which matches the literature that ViT can win on tasks benefiting from global context." 

**Q: "Why did the full PanCAN model fail?"**
> "With only 629 training images, the full PanCAN’s parameter/sample ratio is far too high, which led to NaN losses from epoch 1. This is a scale mismatch: the paper assumes much larger datasets." 

**Q: "Why does multimodal fusion win?"**
> "Combining image + text adds complementary information. In our study, the multimodal late-fusion model reaches 92.40%, outperforming both the best vision-only model (ViT at 86.69%) and the voting ensemble (88.21%)." 

**Q: "How does it compare to Mission 6?"**
> "Mission 6 reached 95.04% with a multimodal approach. Here we reach 92.40% with a lightweight late fusion (EfficientNet + TF‑IDF). The remaining gap suggests there’s still value in richer text encoders or better fusion strategies." 

**Recommendation**: ✅ Well prepared – anticipated FAQ with clear answers

---

### CE3: Complete approach with model comparison ✅ VALIDATED

| Item | Status | Evidence |
|---------|------|---------|
| **Multiple models compared** | ✅ Excellent | 5 evaluated approaches: VGG16, PanCANLite, ViT, ensemble, multimodal (+ full PanCAN failure analysis) |
| **Multiple metrics** | ✅ Excellent | Accuracy, F1, Params, Ratio, Time |
| **Comparative analysis** | ✅ Excellent | Tables + 4-panel charts |
| **End-to-end approach** | ✅ Excellent | Data → Training → Eval → Analysis |

**Complete modeling pipeline**:

```
1. Data Loading & Exploration (Section 4)
   ├─ 7 categories, 1050 images
   ├─ Class distribution (balanced)
   └─ Sample visualization (original vs augmented)

2. Model Architecture (Sections 5–12)
   ├─ Full PanCAN: attempted → NaN losses
   ├─ PanCANLite: lightweight context aggregation baseline
   ├─ VGG16 Baseline: CNN reference
   ├─ ViT-B/16: transformer baseline
   ├─ Soft voting ensemble: ViT + CNNs
   └─ Multimodal fusion: image + text

3. Training & Optimization
   ├─ Full PanCAN → FAILED (NaN losses)
   ├─ VGG16 / PanCANLite / ViT → trained & evaluated
   ├─ Ensemble → evaluated on top of trained models
   └─ Multimodal fusion → evaluated (best performance)

4. Evaluation & Comparison (Results + Conclusions)
   ├─ Vision-only: VGG16 / PanCANLite / ViT
   ├─ Ensemble: soft voting
   ├─ Multimodal: late fusion (best)
   └─ Model efficiency + dataset scale analysis

5. Analysis & Insights (Sections 7–13)
   ├─ Interpretability: saliency + SHAP
   ├─ Literature-driven ensemble and fusion
   └─ Paper constraints vs our dataset scale
```

**Comparative visualizations**:
- ✅ Bar charts: Accuracy, F1-score
- ✅ Log-scale: Parameter counts
- ✅ Efficiency plot: parameter/sample ratios with threshold lines
- ✅ Summary table with all metrics

**Recommendation**: ✅ Excellent – rigorous and complete scientific approach

---

## Overall summary

### ✅ Strengths

| Strength | Impact | Score |
|-------|--------|-------|
| **Strong technology watch** | Recent (2025) paper well leveraged | 5/5 |
| **Working PoC** | Production-ready code with solid results | 5/5 |
| **Technical documentation** | Very complete notebook with math details | 5/5 |
| **Rigorous comparison** | 5 approaches compared with multiple metrics | 5/5 |
| **Critical analysis** | Failures documented and explained | 5/5 |
| **Popularization** | Complex concepts explained clearly | 4/5 |

### ⚠️ Areas to improve

| Gap | Priority | Required action |
|--------|----------|----------------|
| **Formal methodological note** | 🔴 HIGH | Create a PDF following the template (max 10 pages) using the notebook’s final results + XAI figures |
| **Presentation deck** | 🟡 MEDIUM | Create a PowerPoint (max 30 slides) aligned with the notebook narrative (ViT → ensemble → multimodal) |

### Score by competency

```
Competency 1: Technology watch
├─ CE1: Sources            [✅] 5/5
├─ CE2: Math details        [✅] 4/5
└─ CE3: PoC                [✅] 5/5
                           ─────────
                           Score: 93% ✅

Competency 2: Methodological note
├─ CE1: Summary            [⚠️] 3/5  ← Formal note missing
├─ CE2: Metrics            [✅] 5/5
├─ CE3: Interpretability   [✅] 4/5
└─ CE4: Limits             [✅] 5/5
                           ─────────
                           Score: 82% ✅

Competency 3: Oral presentation
├─ CE1: Popularization     [✅] 4/5
├─ CE2: Questions          [✅] 4/5
└─ CE3: Comparison         [✅] 5/5
                           ─────────
                           Score: 87% ✅
```

**Overall score: 88% / 100**

---

## Action plan for full validation

### 🔴 PRIORITY 1 – Blockers (before the defense)

#### Action 1: Methodological Note PDF (4–6h)
```markdown
Create: reports/note_methodologique.pdf

Structure (provided template):
1. Dataset (1 page)
   - 7 Flipkart categories, 1050 images
   - Train/val/test split
   
2. PanCAN concepts (2 pages)
   - Multi-order context with diagrams
   - Cross-scale aggregation
   - Mathematical details (formulas)
   
3. Modeling (2 pages)
   - Vision-only baselines: VGG16, PanCANLite, ViT-B/16
   - Optimization choices (dropout, early stopping, regularization)
   - Metrics: Accuracy, F1-score
   
4. Comparative results (2 pages)
   - Table: VGG16 vs PanCANLite vs ViT vs Ensemble vs Multimodal (+ Full PanCAN failure)
   - Notebook 4-panel charts
   - Conclusion: best result = 92.40% with multimodal fusion; ViT-based methods dominate the final ranking
   
5. Feature Importance (1–2 pages)
   - Saliency maps (ViT)
   - SHAP global + per-class + local explanations
   
6. Limitations & Improvements (1 page)
   - Constraint: 629 samples vs 80K required
   - Gap vs Mission 6 reference (95.04% vs 92.40%)
   - 5 improvement directions
```

**Source**: Sections 4–13 of the notebook to reformat

#### Action 2: Integrate XAI into the methodological note (1–2h)
- Select and embed the figures already produced in the notebook (ViT saliency, global/per-class/local SHAP)
- Add a short “Interpretability” section explaining what each figure demonstrates and what patterns were observed

### 🟡 PRIORITY 2 – Presentation deck (2–3h)

#### Action 3: PowerPoint (30 slides)
```
Suggested structure:

Slides 1-5: Introduction & Context (5 min)
├─ 1. Cover slide
├─ 2. Mission context (technology watch)
├─ 3. Problem: e-commerce classification
├─ 4. Flipkart dataset (7 categories, 1050 images)
└─ 5. Goals: compare CNN vs ViT, ensemble, and multimodal fusion

Slides 6-25: Technology Watch & Experiments (10 min)
├─ 6.  Source paper (arXiv 2025)
├─ 7-8. PanCANLite concepts (diagrams) + why full PanCAN fails here
├─ 9-11. Vision-only baselines (VGG16, PanCANLite, ViT)
├─ 12. Ensemble (soft voting) results
├─ 13. Multimodal fusion results
├─ 14. Interpretability (saliency + SHAP)
└─ 15. Limitations and improvements

Slides 26-30: Conclusions (3 min)
├─ 26. Results summary (92.40% best with multimodal fusion)
├─ 27. Key insights (scaling, param/sample ratio)
├─ 28. Comparison with Mission 6 (multi-modal)
├─ 29. Production recommendations
└─ 30. Thank you / Questions
```

### 🟢 PRIORITY 3 – Optional improvements

#### Action 4: Additional analyses
- Per-class performance breakdown
- Training curves visualization (loss/accuracy)
- Data augmentation impact study
- Hyperparameter sensitivity analysis

---

## Recommended timeline

| Day | Task | Duration | Deliverable |
|------|-------|-------|----------|
| **Day 1 AM** | Methodological note (sections 1–4) | 3h | 8-page PDF |
| **Day 1 PM** | Integrate XAI figures into the note | 2h | Note section + figures |
| **Day 2 AM** | Methodological note (results + conclusions) | 2h | Final tables + discussion |
| **Day 2 PM** | Presentation deck | 3h | 30-slide PPT |
| **Day 3** | Revisions and oral prep | 4h | Defense rehearsal |

**Total effort**: 14h across 3 days

---

## Final checklist before submission

### Required deliverables

- [x] **1. Technology watch notebook** ✅
  - [x] mission8_pancan.ipynb
   - [x] Interpretability & XAI (saliency + SHAP)
  - [x] Complete source code (src/)
  - [x] requirements.txt

- [ ] **2. Methodological note** ❌
  - [ ] ⚠️ **To create**: reports/note_methodologique.pdf
  - [ ] Max 10 pages
  - [ ] Follows the provided template
  - [ ] Includes feature importance

- [ ] **3. Presentation deck** ❌
  - [ ] ⚠️ **To create**: presentation.pptx
  - [ ] Max 30 slides
  - [ ] PoC charts

### Submission naming
```
Mission8_[LastName]_[FirstName].zip
├─ [LastName]_[FirstName]_1_notebook_veille_012026/
│  ├─ mission8_pancan.ipynb
│  ├─ src/
│  ├─ requirements.txt
│  └─ README.md
├─ [LastName]_[FirstName]_2_note_methodologique_012026.pdf
└─ [LastName]_[FirstName]_3_presentation_012026.pptx
```

---

## Defense recommendations

### Timing (30 minutes total)

**Presentation (20 min)**
```
├─ 0–3 min:   Problem recap
├─ 3–10 min:  Models & concepts (PanCANLite, ViT, fusion)
├─ 10–18 min: Results & interpretability
└─ 20 min:    Conclusion
```

**Discussion (5 min)**
- Challenging questions on technical choices
- Justification of parameter/sample ratio
- Multi-modal vs vision-only comparison

**Debrief (5 min)**
- Evaluator feedback
- Strengths / improvement areas

### Key points to defend

1. **Why multi-model watch**: compare CNN, transformer, ensemble, and fusion fairly
2. **Owned failure**: full PanCAN is not feasible with 629 samples (paper assumes 80K+)
3. **Smart adaptation**: PanCANLite designed specifically (97% fewer parameters)
4. **Strong results**: ViT-based methods win (ViT 86.69%, ensemble 88.21%)
5. **Best approach**: multimodal fusion reaches 92.40% (still below Mission 6 reference 95.04%)

### Tricky questions to anticipate

**Q: "Why did ViT beat the CNN baselines here?"**
> "On this dataset, global context seems to matter. ViT-B/16 reaches 86.69% while VGG16 and PanCANLite are at 84.79%. This supports the task-specific point that transformers can win when global dependencies are useful." 

**Q: "Why is multimodal fusion so much better?"**
> "Text contains category clues not always visible in the image. Late fusion combines complementary signals and reaches 92.40%, which is +5.71% over ViT." 

**Q: "Why not combine with Mission 6 (text)?"**
> "We did: the notebook includes a lightweight multimodal fusion (image + text) reaching 92.40%. The next step would be testing richer text encoders or improved fusion to close the gap to the Mission 6 reference." 

---

## Conclusion

### Current status
**88% validated** – excellent technical work, incomplete formal deliverables

### To reach 100% validation
1. 🔴 Create the methodological PDF note (6h)
2. 🟡 Prepare the presentation deck (3h)

**Remaining effort: ~9h across 2–3 days**

### Project strengths
- ✅ Solid and reproducible PoC
- ✅ Rigorous 5-approach comparison
- ✅ Mature critical analysis (failures acknowledged)
- ✅ Production-ready code
- ✅ Excellent technical documentation
- ✅ Interpretability artifacts (saliency + SHAP)

### Main weakness
- ⚠️ Missing formal deliverables (PDF note, presentation)

**Verdict**: Technically very solid project; it mainly requires finalizing the formal deliverables. The core work is strong and demonstrates solid mastery of technology watch practices and advanced modeling.

---

**Generated on**: January 4, 2026  
**Project**: Mission 8 – Technology Watch (ViT / ensemble / multimodal fusion)  
**Status**: In progress – priority actions identified  
**Next step**: finalize the methodological note PDF + presentation
