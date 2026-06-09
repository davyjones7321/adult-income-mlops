import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from datasets import load_dataset

from features import engineer_features, get_X_y, preprocessor

# ── 1. Load data ───────────────────────────────────────────────────────────

def load_data():
    dataset = load_dataset("jlh/uci-adult-income")
    train_df = engineer_features(dataset['train'].to_pandas())
    return train_df

# ── 2. Train one model and log everything to MLflow ───────────────────────

def train_model(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):

        # Build full pipeline: preprocessor + model
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier',   model)
        ])

        # Train
        full_pipeline.fit(X_train, y_train)
        preds = full_pipeline.predict(X_test)
        proba = full_pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc',  auc)
        mlflow.sklearn.log_model(full_pipeline, artifact_path='model')

        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"F1:      {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(classification_report(y_test, preds))

        return full_pipeline, auc

# ── 3. Run all models and pick the best ───────────────────────────────────

if __name__ == "__main__":
    
    # Load and split
    df = load_data()
    X, y = get_X_y(df)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Models to compare
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=100),
    }

    mlflow.set_experiment("adult-income-classifier")

    results = {}
    for name, model in models.items():
        pipeline, auc = train_model(model, name, X_train, y_train, X_test, y_test)
        results[name] = auc

    # Print winner
    best = max(results, key=results.get)
    print(f"\n🏆 Best model: {best} with AUC-ROC: {results[best]:.4f}")