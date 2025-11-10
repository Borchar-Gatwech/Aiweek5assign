📘 Overview

This project implements an AI system to predict the likelihood of patient readmission within 30 days of discharge.
It follows the full AI Development Workflow — from problem definition to deployment — as part of the PLP Academy AI Assignment.

The application is built using Python, FastAPI, and Scikit-learn, and demonstrates the complete process of:

Data preprocessing

Model training & evaluation

Model saving/loading

Simple API + web interface for predictions

🚀 Features

🧠 Logistic Regression model trained on simulated hospital data

⚙️ Scaler and model persisted using joblib

🌐 FastAPI backend serving predictions

💻 Simple HTML frontend (form-based interface)

🔍 Interactive API docs via Swagger UI (/docs)

🗂️ Project Structure
Aiweek5assign/
│
├── app.py                     # FastAPI web app (API + HTML frontend)
├── hospital_readmission_model.ipynb          # Notebook used to train and save the model
├── readmission_model.pkl      # Saved Logistic Regression model
├── scaler.pkl                 # StandardScaler used for feature normalization
├── templates/
│   └── index.html             # Web form for user input
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Rufron/Aiweek5assign.git
cd Aiweek5assign

2️⃣ Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # On Windows
# or
source venv/bin/activate     # On Mac/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the FastAPI app
python -m uvicorn app:app --reload

5️⃣ Open in your browser

🌍 Web App: http://127.0.0.1:8000/

📘 API Docs (Swagger UI): http://127.0.0.1:8000/docs

🧠 Example Prediction

You can either use the HTML form or call the API directly.

Sample API Request:

curl -X POST "http://127.0.0.1:8000/predict?age=60&num_prior_admissions=3&days_since_last_admission=90&avg_lab_result=110.5&comorbidity_score=2&length_of_stay=4"


Response:

{"readmission_risk": "High"}

🧩 Model Development Summary

Model Used: Logistic Regression

Features:

Age

Number of prior admissions

Days since last admission

Average lab result

Comorbidity score

Length of stay

Training: Simulated dataset (generated in hospital_readmission_model.ipynb)

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

⚖️ Ethical & Privacy Considerations

🔒 Patient data privacy is preserved — only synthetic data is used.

⚖️ Potential model bias and fairness issues are discussed in the report.

🩺 Compliance considerations include HIPAA and data governance policies.

💡 Future Improvements

Integration with real hospital EHR systems

Model retraining pipeline for handling concept drift

Improved interpretability using SHAP or LIME

Cloud deployment (e.g., Render, Hugging Face Spaces)

👨‍💻 Contributors

Brian Kipchumba
Borchar Gatwetch