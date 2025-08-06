# Selective Data Clarity Model (SDC-AF)




The **Selective Data Clarity Model (SDC-AF)** is a Python-based preprocessing framework for automated feature selection and outlier elimination in machine learning workflows.

This model enhances data clarity, improves model performance, and maintains feature interpretability, making it suitable for real-world domains such as healthcare, finance, and scientific research.

---

## Key Features

- Task-aware feature scoring:
  - Mutual Information (for classification)
  - Pearson Correlation (for regression)
  - Random Forest Importance (model-based)
- Z-Score-based outlier detection
- Low-variance feature elimination
- Confidence-threshold-based feature selection
- Retention of original feature names to preserve interpretability

---

## Performance Highlights

| Dataset      | Preprocessing Method | Accuracy |
|--------------|----------------------|----------|
| Iris         | PCA                  | 95%      |
| Iris         | SDC Model            | 100%     |

- Feature reduction: 30–50%
- Outlier removal: 3–7% of rows
- Accuracy improvement: up to 5%






