# from fastapi import FastAPI
# import joblib
# import numpy as np

# # Load saved model and scaler
# model = joblib.load("readmission_model.pkl")
# scaler = joblib.load("scaler.pkl")

# app = FastAPI(title="Hospital Readmission Predictor")

# @app.get("/")
# def root():
#     return {"message": "Hospital Readmission API Running"}

# @app.post("/predict")
# def predict(age: int, num_prior_admissions: int, days_since_last_admission: int,
#             avg_lab_result: float, comorbidity_score: int, length_of_stay: int):

#     X = np.array([[age, num_prior_admissions, days_since_last_admission,
#                    avg_lab_result, comorbidity_score, length_of_stay]])

#     X_scaled = scaler.transform(X)
#     pred = model.predict(X_scaled)[0]

#     return {"readmission_risk": "High" if pred == 1 else "Low"}


from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("readmission_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Hospital Readmission Predictor")

# Setup template directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
    request: Request,
    age: int = Form(...),
    num_prior_admissions: int = Form(...),
    days_since_last_admission: int = Form(...),
    avg_lab_result: float = Form(...),
    comorbidity_score: int = Form(...),
    length_of_stay: int = Form(...)
):
    # Prepare input for model
    X = np.array([[age, num_prior_admissions, days_since_last_admission,
                   avg_lab_result, comorbidity_score, length_of_stay]])
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    result = "High" if pred == 1 else "Low"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result}
    )
