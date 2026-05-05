from xgboost import XGBClassifier
import joblib
#

def load_everything():
    model = XGBClassifier()
    model.load_model("model.ubj")

    X_train = joblib.load("X_train.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    feature_names = joblib.load("feature_names.pkl")

    return model, X_test, y_test, X_train, feature_names

model, X_test, y_test, X_train, feature_names = load_everything()


