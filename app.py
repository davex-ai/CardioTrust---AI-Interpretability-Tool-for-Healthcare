import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from lime import lime_tabular
from xgboost import XGBClassifier

# 1. Cache the model so it doesn't run 20 trials/retrain on every slider move
@st.cache_resource
def load_everything():
    model = XGBClassifier()
    model.load_model("model.ubj")

    X_train = joblib.load("X_train.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    feature_names = joblib.load("feature_names.pkl")

    return model, X_test, y_test, X_train, feature_names

model, X_test, y_test, X_train, feature_names = load_everything()

st.title("🫀 CardioTrust: Explainable AI")

# 2. Sidebar: Patient Input
st.sidebar.header("Patient Vitals")

# Numeric Inputs
age = st.sidebar.slider("Age", 20, 80, 50)
trestbps = st.sidebar.slider("Resting BP", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
thalch = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

# Categorical Inputs (Simplified for user)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
fbs = st.sidebar.checkbox("Fasting Blood Sugar > 120 mg/dl")
restecg = st.sidebar.selectbox("Resting ECG", ["normal", "st-t abnormality", "lv hypertrophy"])
exang = st.sidebar.checkbox("Exercise Induced Angina")
slope = st.sidebar.selectbox("Peak Exercise Slope", ["upsloping", "flat", "downsloping"])

# 3. Construct Input DataFrame (Must match X_train columns exactly)
input_data = {
    'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch, 'oldpeak': oldpeak,
    'sex_Male': 1 if sex == "Male" else 0,
    'cp_atypical angina': 1 if cp == "atypical angina" else 0,
    'cp_non-anginal': 1 if cp == "non-anginal" else 0,
    'cp_typical angina': 1 if cp == "typical angina" else 0,
    'fbs_True': 1 if fbs else 0,
    'fbs_Unknown': 0,
    'restecg_lv hypertrophy': 1 if restecg == "lv hypertrophy" else 0,
    'restecg_normal': 1 if restecg == "normal" else 0,
    'restecg_st-t abnormality': 1 if restecg == "st-t abnormality" else 0,
    'exang_True': 1 if exang else 0,
    'exang_Unknown': 0,
    'slope_downsloping': 1 if slope == "downsloping" else 0,
    'slope_flat': 1 if slope == "flat" else 0,
    'slope_upsloping': 1 if slope == "upsloping" else 0,
}
input_df = pd.DataFrame([input_data])

# 4. Main Page: Prediction
prediction_prob = model.predict_proba(input_df)[0][1]
st.metric("Heart Disease Risk", f"{prediction_prob:.2%}")

if prediction_prob > 0.5:
    st.error("High Risk Detected")
else:
    st.success("Low Risk Detected")

# 5. EXPLANATION LAYER
st.subheader("Why this prediction?")
tab1, tab2 = st.tabs(["SHAP (Feature Impact)", "LIME (Local Proxy)"])

with tab1:
    st.write("### SHAP Waterfall Plot")
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    # Explain the specific user input from the sidebar
    local_shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(local_shap_values[0], show=False)
    st.pyplot(fig)
    plt.clf()

with tab2:
    st.write("### LIME Explanation")
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=['Healthy', 'Heart Disease'],
        mode='classification'
    )

    exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0].values,
        predict_fn=model.predict_proba
    )
    with open("lime.html", "w", encoding="utf-8") as f:
        f.write(exp.as_html())

    st.iframe("lime.html", height=800)