
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score, \
    confusion_matrix, roc_auc_score
import pandas as pd
from xgboost import XGBClassifier
from load_data import df_c
import optuna

def prepare(df, reference_columns=None):
    df = pd.get_dummies(df, drop_first=True)

    if reference_columns is not None:
        df = df.reindex(columns=reference_columns, fill_value=0)

    return df

def objective(trial, X, y):
    # params = {
    #     'max_depth': trial.suggest_int('max_depth', 3, 10),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
    #     'n_estimators': trial.suggest_int('n_estimators', 100, 600),
    #     'subsample': trial.suggest_float('subsample', 0.7, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
    #     'random_state': 42,
    #     'eval_metric': 'logloss'
    #
    # }
    params = {
        'max_depth': 3,
        'learning_rate': 0.010081523953294461,
        'n_estimators': 477,
        'subsample': 0.998437874354703,
        'colsample_bytree': 0.8155957977429815,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)
    score = cross_val_score(model, X, y, n_jobs=-1, cv=3, scoring='accuracy').mean()
    return score

def run_experiment(df, name):
    df = prepare(df)

    X = df.drop('num', axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Class counts in y:")
    print(y.value_counts())

    # 2. Run Bayesian Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)

    best_params = study.best_params
    final_model = XGBClassifier(**best_params, early_stopping_rounds=20, random_state=42)
    X_sub_train, X_val, y_sub_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    final_model.fit(X_sub_train, y_sub_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Evaluation
    preds = final_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    prec_score = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds)
    probs = final_model.predict_proba(X_test)[:, 1]

    feat_importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # 5. Logging
    log_entry = (
        f"\n{'=' * 40}\n"
        f"Experiment: {name} | Model: XGBOOST\n"
        f"Best Params: {best_params}\n"
        f"ROC AUC SCORE: {roc_auc_score(y_test, probs)}\n"

        f"Confusion Matrix: {confusion_matrix(y_test, preds)}\n"
        f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | Recall: {recall:.4f} | Precision: {prec_score:.4f}\n"
        f"Important Features:\n{feat_importances.head(10)}\n"
        f"Full Report:\n{report}\n"
    )

    with open("model_results.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

    print(f"Experiment {name} completed. Results saved.")


def run_final_model(df, name):
    df = prepare(df)

    X = df.drop('num', axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Use the Best Params from your logs
    best_params = {
        'max_depth': 3,
        'learning_rate': 0.010081523953294461,
        'n_estimators': 477,
        'subsample': 0.998437874354703,
        'colsample_bytree': 0.8155957977429815,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    # 2. Initialize and Train
    final_model = XGBClassifier(**best_params, early_stopping_rounds=20)

    # Validation split for early stopping
    X_sub_train, X_val, y_sub_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    final_model.fit(X_sub_train, y_sub_train, eval_set=[(X_val, y_val)], verbose=False)

    # 3. Evaluation
    preds = final_model.predict(X_test)
    probs = final_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    roc_auc = roc_auc_score(y_test, probs)

    print(f"Final Model: {name}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {roc_auc:.4f}")
    final_model.save_model("model.ubj")
    return final_model


df_c_tuning = df_c.drop("dataset", axis=1)
run_experiment(df_c_tuning, "Dataset C (drop ca, thal)")
# run_final_model(df_c_tuning, "Dataset C (drop ca, thal)")