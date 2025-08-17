# Selective Data Clarity Model (SDC-AF)

[Selective Data Clarity Model - Research Paper (Online View)](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2FAdnanThange%2FAI-ML-Portfolio%2Frefs%2Fheads%2Fmain%2FResearch_Paper%2FSelectiveDataClarityMode%2FSelective%2520Data%2520Clarity%2520Model_Research_Paper_updated.docx&wdOrigin=BROWSELINK)










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

| Dataset             | Preprocessing Method | Accuracy / R² Score |
|---------------------|----------------------|--------------------|
| Iris                | PCA                  | 95%                |
| Iris                | SDC Model            | 100%               |
| Student Performance | PCA                  | 0.98898            |
| Student Performance | SDC Model            | 0.9890             |



- Feature reduction: 30–50%
- Outlier removal: 3–7% of rows
- Accuracy improvement: up to 5%






