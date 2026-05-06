# Model Card for Heart Disease Prediction with Explainable AI

<!-- Quick summary -->

This model predicts the presence of heart disease using clinical features and provides SHAP-based explanations to improve interpretability and trust in predictions.

---

## Model Details

### Model Description

This project is an interpretable machine learning system for heart disease prediction. It uses an XGBoost classifier optimized with Bayesian hyperparameter tuning (Optuna). The model is paired with SHAP to explain individual predictions and global feature importance, focusing on building trust in high-stakes healthcare predictions.

- **Developed by:** Dave (Independent Student Developer)
- **Funded by:** None
- **Shared by:** Dave
- **Model type:** Gradient Boosted Decision Tree (XGBoost)
- **Language(s) (NLP):** N/A (Tabular ML)
- **License:** MIT
- **Finetuned from model:** XGBoost base implementation

---

### Model Sources

- **Repository:** https://github.com/davex-ai/CardioTrust---AI-Interpretability-Tool-for-Healthcare
- **Demo:** Hugging Face Space (Streamlit App)

---

## Uses

### Direct Use

This model can be used to:
- Predict risk of heart disease from patient clinical data
- Explain predictions using SHAP values
- Demonstrate interpretable ML systems in healthcare

### Downstream Use

- Clinical decision support prototypes
- Medical AI education tools
- Explainable AI research demonstrations

### Out-of-Scope Use

- Not intended for real clinical diagnosis
- Not a substitute for medical professionals
- Not validated for deployment in hospitals without further clinical testing

---

## Bias, Risks, and Limitations

- Dataset is limited in size and demographic diversity
- Model may over-rely on correlated clinical features
- False negatives are critical risk (missed disease cases)
- Performance may degrade outside training distribution

### Recommendations

- Do not use for medical decisions without expert validation
- Always pair predictions with explainability (SHAP analysis)
- Threshold tuning is recommended for high-recall settings in healthcare

---

## How to Get Started with the Model

```python
from xgboost import XGBClassifier
import shap

model = XGBClassifier()
model.load_model("model.ubj")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)
```
# Training Details

## Training Data

The model is trained on a processed heart disease dataset with:

- Categorical encoding (one-hot encoding)
- Missing value handling (median imputation / "Unknown" category)
- Feature engineering on clinical variables

## Training Procedure

- Model: XGBoost Classifier
- Hyperparameter tuning: Optuna (Bayesian Optimization)
- Evaluation: Train/test split (80/20 stratified)

## Training Hyperparameters

- max_depth: 3–10 (optimized)
- learning_rate: 0.01–0.2
- n_estimators: 100–600
- subsample: 0.7–1.0
- colsample_bytree: 0.7–1.0
- evaluation metric: logloss

---

# Evaluation

## Testing Data & Metrics

### Testing Data
Held-out 20% stratified test split from dataset

### Metrics
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC (important for medical risk ranking)

## Results

- Accuracy: ~0.85
- ROC AUC: ~0.90+
- Strong recall for positive class (heart disease cases)

---

# Summary

The model performs well on classification and ranking tasks but requires interpretability tools (SHAP) due to medical risk sensitivity.

---

# Model Examination

SHAP is used to:

- Explain individual predictions (local interpretability)
- Show global feature importance
- Analyze failure cases (false positives and false negatives)

## Key insight:

- Model strongly relies on features like `exang_True`, `cp_type`, and `oldpeak`
- False negatives occur when patients have “healthy-looking” vitals but hidden risk

---

# Environmental Impact

- Hardware Type: CPU (local training)
- Hours used: ~2–5 hours (experimental training)
- Cloud Provider: None (local machine)
- Compute Region: N/A
- Carbon Emitted: Minimal (small-scale training)

---

# Technical Specifications

## Model Architecture

XGBoost Gradient Boosted Trees optimized for tabular structured medical data.

## Compute Infrastructure

- Local development machine
- Python environment with scikit-learn, xgboost, shap, optuna
