import joblib
import pandas as pd


def load_model(path="model/saved/model.pkl"):
    return joblib.load(path)


def predict(model, input_dict: dict):
    df = pd.DataFrame([input_dict])
    preds = model.predict(df)

    return {
        "cycle": float(preds[0][0]),
        "bone": float(preds[0][1]),
        "cortisol": float(preds[0][2]),
    }