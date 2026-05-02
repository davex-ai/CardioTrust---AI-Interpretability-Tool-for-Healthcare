
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from load_data import df_a, df_b, df_c


def prepare(df, reference_columns=None):
    df = pd.get_dummies(df, drop_first=True)

    if reference_columns is not None:
        df = df.reindex(columns=reference_columns, fill_value=0)

    return df

def run_experiment(df, name, model_type="rf"):
    df = prepare(df)

    X = df.drop('num', axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    feat_name = X.columns
    if model_type == "lr":
        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
    elif model_type == "xgb":
        model = XGBClassifier(random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    if model_type == "lr":
        importances = abs(model.coef_[0])
    else:
        importances = model.feature_importances_

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    f1 = f1_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds)
    feat_importances = pd.Series(importances, feat_name).sort_values(ascending=False)

    # Prepare log string
    log_entry = (
        f"\n{'=' * 40}\n"
        f"Experiment: {name} | Model: {model_type.upper()}\n"
        f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | Recall: {recall:.4f}\n"
        f"Important Features: {feat_importances}"
        f"Full Report:\n{report}\n"
    )

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

    return acc

run_experiment(df_a.copy(), "Dataset A (raw zeros)", "lr")
run_experiment(df_a.copy(), "Dataset A (raw zeros)", "xgb")
run_experiment(df_a.copy(), "Dataset A (raw zeros)")

run_experiment(df_b.copy(), "Dataset B (zero→NaN)", "lr" )
run_experiment(df_b.copy(), "Dataset B (zero→NaN)", "xgb")
run_experiment(df_b.copy(), "Dataset B (zero→NaN)")

run_experiment(df_c.copy(), "Dataset C (drop ca, thal)", "lr" )
run_experiment(df_c.copy(), "Dataset C (drop ca, thal)", "xgb" )
run_experiment(df_c.copy(), "Dataset C (drop ca, thal)")