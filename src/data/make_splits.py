import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fake-job-detector")


# src/data/make_splits.py

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/fake_job_postings.csv"  # confirm the exact filename
TARGET = "fraudulent"

TEXT_FIELDS = ["title","company_profile","description","requirements","benefits"]
META_FIELDS = ["telecommuting","has_company_logo","has_questions","employment_type",
               "required_experience","required_education","industry","function"]

def main():
    df = pd.read_csv(RAW_PATH)
    df = df.drop_duplicates(subset=["title","company_profile","description","requirements"], keep="first")
    for col in TEXT_FIELDS:
        df[col] = df[col].fillna("")
    for col in META_FIELDS:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    assert TARGET in df.columns, f"Missing target column: {TARGET}"
    df[TARGET] = df[TARGET].astype(int)

    train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=42)
    valid, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=42)

    train.to_csv("data/processed/train.csv", index=False)
    valid.to_csv("data/processed/valid.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("Wrote data/processed/train.csv, valid.csv, test.csv")

if __name__ == "__main__":
    main()
