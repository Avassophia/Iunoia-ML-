# Iunoia Core — Prediction Engine

## Overview

This project provides a hybrid prediction system for estimating:

* **Cortisol Load**
* **Cycle Variability**
* **Bone Loss Rate**

It combines:

1. **Rule-based physiological simulator** (default fallback)
2. **Machine Learning model (Linear Regression)** with:

   * Min-Max normalization (numerical features)
   * One-hot encoding (categorical features)

---

## Project Structure

```
iunoia-core/
│
├── data/
│   ├── raw/
│   │   └── your_data.csv
│   └── processed/
│
├── model/
│   ├── features.py          # Feature engineering
│   ├── predictor.py         # Main prediction interface (hybrid system)
│   ├── preprocess.py        # Preprocessing (scaling + encoding)
│   ├── train.py             # Model training pipeline
│   ├── inference.py         # Model loading + prediction
│   └── saved/
│       └── model.pkl        # Trained model
│
├── demo/
│   └── sample_input.json
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

### Targets

The model predicts:

* `cycle`
* `bone`
* `cortisol`

### Features

All remaining columns are used as input features.

### Preprocessing

* **Numerical features** → MinMaxScaler
* **Categorical features** → OneHotEncoder

### Model

* Multi-output **Linear Regression**
* Implemented via `sklearn.pipeline.Pipeline`

---

## Setup

### Install dependencies

```
pip install -r requirements.txt
```

---

## Training the Model

1. Place your dataset in:

```
data/raw/your_data.csv
```

2. Ensure it contains:

* Input features (mixed numerical + categorical)
* Target columns:

  * `cycle`
  * `bone`
  * `cortisol`

3. Run training:

```
python -m model.train
```

This will:

* Train the model
* Evaluate performance (MSE, R²)
* Save model to:

```
model/saved/model.pkl
```

---

## Running Predictions

### Option 1 — Full pipeline (recommended)

```
python -m model.predictor
```

### Option 2 — Direct ML inference

```python
from model.inference import load_model, predict

model = load_model()

input_data = {
    "stress_score": 0.7,
    "sleep_deficit": 0.4,
    "gravity_factor": 1,
    ...
}

result = predict(model, input_data)
print(result)
```

---

## Hybrid Prediction System

`predictor.py` automatically chooses:

| Condition              | Behavior                |
| ---------------------- | ----------------------- |
| ML model available     | Uses trained model      |
| ML model missing/fails | Falls back to simulator |

This ensures:

* Reliability (no hard failures)
* Backward compatibility
* Smooth transition to ML

---

## Data Requirements

Your dataset should:

* Contain **no missing values** (or be pre-cleaned)
* Include both:

  * Numerical columns (floats/ints)
  * Categorical columns (strings)
* Include target columns:

  ```
  cycle, bone, cortisol
  ```

---

## Example Workflow

1. Add dataset → `data/raw/`
2. Train model → `python -m model.train`
3. Run predictions → `python -m model.predictor`

---

## Future Improvements

* Replace Linear Regression with:

  * Ridge / Lasso
  * Random Forest
  * Gradient Boosting
* Add feature importance analysis
* Integrate experiment tracking (MLflow)
* Deploy as API endpoint

---

## Notes

* The simulator in `predictor.py` is still active as a fallback
* The ML model overrides simulator outputs when available
* Designed for easy upgrade to cloud deployment (e.g., Azure ML)

---

## License

Internal / TBD
