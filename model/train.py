import pandas as pd
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from .preprocess import build_preprocessor
from .config_loader import load_constants


CONST = load_constants()
TARGETS = CONST["ml"]["target_columns"]
TEST_SIZE = CONST["ml"]["test_size"]
RANDOM_STATE = CONST["ml"]["random_state"]


def train(csv_path, model_out="model/saved/model.pkl"):
    df = pd.read_csv(csv_path)
    df = df.dropna()

    X = df.drop(columns=TARGETS)
    y = df[TARGETS]

    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    Path("model/saved").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)

    print(f"Model saved → {model_out}")


if __name__ == "__main__":
    train("data/raw/your_data.csv")