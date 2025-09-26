import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fake-job-detector")


# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="Fake Job Posting Detector")

# Allow the Streamlit UI (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("assets/baseline_tfidf_lr.joblib")

class JobIn(BaseModel):
    title: str = ""
    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "title": "Data Entry Clerk",
                "company_profile": "Small remote team",
                "description": "Work from home, weekly gift card payments",
                "requirements": "Message recruiter on Telegram",
                "benefits": "Flexible hours"
            }]
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(job: JobIn):
    text = " ".join([job.title, job.company_profile, job.description, job.requirements, job.benefits])
    proba = float(model.predict_proba([text])[0][1])
    label = int(proba >= 0.5)
    risk_kw = ["gift card", "telegram", "no experience", "wire money", "crypto", "whatsapp"]
    flags = [kw for kw in risk_kw if kw in text.lower()]
    return {"label": label, "probability": proba, "threshold": 0.5, "risk_flags": flags, "model_version": "tfidf_lr_v1"}
