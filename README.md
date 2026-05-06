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
