
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder

def create_titanic_features(n_samples=800, seed=42):
    """Generate synthetic Titanic-like training data."""
    np.random.seed(seed)
    data = pd.DataFrame({
        "Pclass": np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.30, 0.45]),
        "Age": np.random.normal(30, 12, n_samples).clip(1, 80),
        "SibSp": np.random.choice([0, 1, 2, 3], n_samples, p=[0.60, 0.25, 0.10, 0.05]),
        "Parch": np.random.choice([0, 1, 2], n_samples, p=[0.70, 0.20, 0.10]),
        "Fare": np.random.exponential(30, n_samples).clip(5, 300),
        "Sex_enc": np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        "Embarked_enc": np.random.choice([0, 1, 2], n_samples, p=[0.70, 0.20, 0.10]),
    })
    data["Survived"] = (
        (data["Sex_enc"] == 1) * 0.4 +
        (data["Pclass"] == 1) * 0.3 +
        np.random.normal(0, 0.1, n_samples)
    ).clip(0, 1).round().astype(int)
    return data

def train():
    mlflow.set_experiment("titanic-production-model")
    data = create_titanic_features()
    feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_enc", "Embarked_enc"]
    X = data[feature_cols]
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"n_estimators": 100, "max_depth": 8, "min_samples_split": 2, "random_state": 42}

    with mlflow.start_run(run_name="RandomForest-Production"):
        mlflow.log_params(params)
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "auc": round(roc_auc_score(y_test, y_prob), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # save model locally for FastAPI
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # save reference data for monitoring
        X_train_ref = X_train.copy()
        X_train_ref["Survived"] = y_train.values
        X_train_ref.to_csv("artifacts/reference_data.csv", index=False)

        print(f"Training complete!")
        print(f"Accuracy: {metrics['accuracy']} | AUC: {metrics['auc']} | F1: {metrics['f1']}")
        print(f"Model saved to artifacts/model.pkl")
        print(f"Reference data saved to artifacts/reference_data.csv")
        return metrics

if __name__ == "__main__":
    train()
