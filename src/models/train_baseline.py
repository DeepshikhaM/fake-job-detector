# src/models/train_baseline.py
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# --- MLflow setup ---
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")         # MLflow server/UI
mlflow.set_experiment("fake-job-detector")               # experiment name

TARGET = "fraudulent"
TEXT_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]

def concat_text(df):
    parts = [df[c].fillna("") if c in df.columns else "" for c in TEXT_FIELDS]
    return parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " " + parts[4]

def main():
    train = pd.read_csv("data/processed/train.csv")
    valid = pd.read_csv("data/processed/valid.csv")

    X_train, y_train = concat_text(train), train[TARGET].astype(int)
    X_valid, y_valid = concat_text(valid), valid[TARGET].astype(int)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000)),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")),
    ])

    # Optional: capture many sklearn details automatically
    # mlflow.sklearn.autolog()  # uncomment if you prefer autologging

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_cv = cross_val_score(pipe, X_train, y_train, scoring="f1", cv=cv)

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_valid)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc = float(roc_auc_score(y_valid, proba))
    rpt = classification_report(y_valid, preds, output_dict=True)

    # --- MLflow logging block ---
    with mlflow.start_run(run_name="tfidf_lr_baseline_v1"):
        # params
        mlflow.log_param("vectorizer", "tfidf")
        mlflow.log_param("ngram_range", "1-2")
        mlflow.log_param("min_df", 3)
        mlflow.log_param("max_features", 200000)
        mlflow.log_param("classifier", "logreg_lbfgs_balanced")
        mlflow.log_param("cv_splits", 5)
        mlflow.log_param("seed", 42)

        # metrics
        mlflow.log_metric("cv_f1_mean", float(f1_cv.mean()))
        mlflow.log_metric("cv_f1_std", float(f1_cv.std()))
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("f1_weighted", float(rpt["weighted avg"]["f1-score"]))
        mlflow.log_metric("precision_weighted", float(rpt["weighted avg"]["precision"]))
        mlflow.log_metric("recall_weighted", float(rpt["weighted avg"]["recall"]))

        # confusion matrix figure
        disp = ConfusionMatrixDisplay.from_predictions(y_valid, preds)
        plt.title("Validation Confusion Matrix")
        mlflow.log_figure(disp.figure_, "eval/confusion_matrix_valid.png")

        # model artifact (MLflow-managed)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        # optional: text report/artifact
        mlflow.log_text(classification_report(y_valid, preds), "eval/classification_report.txt")

    # Local artifact (outside MLflow, same as before)
    Path("assets").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, "assets/baseline_tfidf_lr.joblib")
    print("Saved model to assets/baseline_tfidf_lr.joblib")

    # Console printouts remain the same
    print("CV F1:", f1_cv.mean(), "+/-", f1_cv.std())
    print("ROC-AUC:", roc)
    print(classification_report(y_valid, preds))
    print("Confusion matrix:\n", confusion_matrix(y_valid, preds))

if __name__ == "__main__":
    main()
