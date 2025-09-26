# ui/app.py

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fake-job-detector")


import streamlit as st, requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("Fake Job Posting Detector")

title = st.text_input("Title")
company = st.text_area("Company profile")
desc = st.text_area("Description")
reqs = st.text_area("Requirements")
benef = st.text_area("Benefits")

if st.button("Predict"):
    payload = {
        "title": title,
        "company_profile": company,
        "description": desc,
        "requirements": reqs,
        "benefits": benef
    }
    r = requests.post(API_URL, json=payload, timeout=30)
    st.json(r.json())
