# Chest-XRay-Pathology-Detection-XAI
Multi-Label Chest X-Ray Pathology Detection on NIH ChestX-ray14 using DenseNet121 with Grad-CAM &amp; SHAP Explainability
# Chest X-Ray Pathology Detection with Explainable AI (XAI)

Multi-label thoracic pathology classification across 14 diseases on the **NIH ChestX-ray14** dataset using a fine-tuned **DenseNet121** backbone, **patient-isolated splitting**, **Youden's $J$ threshold calibration**, and dual **Grad-CAM + SHAP** visual explainability.

## Pipeline Overview
1. **Streaming Subset Curation:** Quota-balanced sampling from NIH ChestX-ray14 (`250` max per pathology, `500` max `No Finding`) resized to `224x224`.
2. **Multi-Hot Encoding:** 14-dimensional independent binary targets (`[0, ..., 0]` for `No Finding`).
3. **Patient-Isolated Splits:** Two-stage `GroupShuffleSplit` on `patient_id` (70% Train / 15% Val / 15% Test) with zero patient overlap.
4. **Model & Imbalance Loss:** ImageNet-pretrained `DenseNet121` trained with inverse-frequency `BCEWithLogitsLoss(pos_weight)` and FP16 mixed precision.
5. **Explainable AI & Error Auditing:** Spatial localization via **Grad-CAM**, pixel-level attribution via **SHAP (`PartitionExplainer`)**, and high-confidence false-positive shortcut auditing.

## Key Results (Held-Out Patient-Isolated Test Set)
* **Macro Test ROC-AUC:** `[Paste your Mean Test AUC from Cell 16]`
* **Macro Test F1-Score:** `[Paste your Mean Test F1 from Cell 16]`
* **Macro Test Recall:** `[Paste your Mean Test Recall from Cell 16]`

Full per-pathology metrics are available in [`results/test_metrics.csv`](results/test_metrics.csv) and [`results/literature_comparison.csv`](results/literature_comparison.csv).

## Visual Results

### Training Convergence & 14-Pathology ROC Curves
![ROC and Training Curves](figures/roc_and_training_curves.png)

### Grad-CAM Anatomical Localization
![Grad-CAM Explanations](figures/gradcam_explanations.png)

### SHAP Pixel-Level Attribution
![SHAP Explanations](figures/shap_explanations.png)

### False-Positive Shortcut Audit
![False Positive Grad-CAM](figures/false_positive_gradcam.png)

### Published Baseline Comparison (Wang et al. 2017 & CheXNet)
![Baseline Comparison](figures/baseline_comparison_chart.png)

## Repository Structure
* `notebooks/Chest_XRay_Pathology_Detection_XAI.ipynb` — Complete end-to-end Google Colab implementation.
* `results/` — Per-class test metrics, false-positive audit logs, and published benchmark comparisons.
* `figures/` — High-resolution (300 DPI) evaluation and XAI plots.
