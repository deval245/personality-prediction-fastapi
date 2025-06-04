
Personality Prediction App (FastAPI + PyTorch + Ngrok)

This project is a personality type prediction system built using **FastAPI**, **PyTorch**, and deployed via **Ngrok**. It uses behavioral and social inputs to classify users as either an **Introvert** or **Extrovert**.

---

## 🚀 Features

- ML model trained with PyTorch (`.pth` file)
- FastAPI backend for serving predictions
- Swagger UI (`/docs`) for testing the API
- Ngrok tunneling for public URL access
- Clean JSON response with:
  - Predicted personality type
  - Confidence score
  - Explanation message
- Dropdowns and predefined inputs supported in API schema

---

## 📁 Folder Structure
personality-prediction-fastapi/
│
├── app/
│ ├── main.py # Entry point (ngrok + FastAPI server)
│ ├── api.py # FastAPI app setup
│ ├── routes.py # API routes
│ ├── model_loader.py # Loads and applies the ML model
│ ├── schema.py # Request & response models
│ ├── utils.py # Utilities (e.g., explanation logic)
│
├── models/
│ └── personality_model.pth # Trained PyTorch model (excluded via .gitignore)
│
├── .env # Environment variables (e.g., NGROK_AUTH_TOKEN)
├── .gitignore
├── requirements.txt
└── README.md

---

## 📦 Setup & Run (Colab Friendly)

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   NGROK_AUTH_TOKEN=your-token-here
#for run
python3 -m app.main

#example input
{
  "Time_spent_Alone": 24,
  "Social_event_attendance": 0,
  "Going_outside": 0,
  "Friends_circle_size": 2,
  "Post_frequency": 0,
  "Stage_fear_Yes": 1,
  "Drained_after_socializing_Yes": 1
}
response
{
  "predicted_personality": "Introvert",
  "confidence_score": 0.87,
  "explanation": "You seem to enjoy solitude and have a smaller social circle."
}
🧠 Tech Stack
FastAPI

PyTorch

Ngrok

Uvicorn

pydantic

🔒 Security Notes
Tokens are managed via .env and excluded in .gitignore.

Avoid pushing .pth or model binaries to public repos.

📌 To Do
Add frontend UI (e.g., Streamlit or React)

Train on more diverse datasets

Add model versioning with MLflow




