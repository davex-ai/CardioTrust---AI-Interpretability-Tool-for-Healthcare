import shap
import xgboost as xgb
from model_tuning import df_c_tuning, run_final_model

# 1. Load model and data (Ensure X_test has the same columns as training)
model, X_test, y_test, X_train = run_final_model(df_c_tuning, "Final Model")
# model = xgb.XGBClassifier()
model.load_model("model.ubj")

# 2. Initialize SHAP Explainer
explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
#
# # explainer = shap.Explainer(model)
# # shap_values = explainer(X_test)
#
# # 3. Summary Plot
# shap.summary_plot(shap_values, X_test)



# Choose a specific patient (e.g., index 10)
patient_idx = 10
patient_data = X_test.iloc[patient_idx : patient_idx + 1]

# Generate local SHAP values
local_shap_values = explainer(patient_data)

# Waterfall plot: Excellent for Streamlit
shap.plots.waterfall(local_shap_values[0])




from lime import lime_tabular

# 1. Setup LIME
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Healthy', 'Heart Disease'],
    mode='classification'
)

# 2. Explain the same patient
exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[patient_idx].values,
    predict_fn=model.predict_proba
)

exp.save_to_file('lime_result.html')
print("LIME explanation saved! Open 'lime_result.html' in your browser.")