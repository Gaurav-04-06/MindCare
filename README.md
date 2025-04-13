# 🧠 MindCare: Mental Health Prediction App (Streamlit + Docker)

MindCare is a machine learning-based mental health prediction web application built using Streamlit. It utilizes a trained RandomForest model and optionally integrates with MLflow for experiment tracking. This Dockerized version allows you to easily build and run the app anywhere.

---

## 🚀 Features

- Predict mental health risk based on user input
- Intuitive and lightweight UI built with Streamlit
- Dockerized for platform-independent deployment
- MLflow integration for experiment tracking (optional)

---

## 🗂 Project Structure

mindcare/
├── app.py               # Main Streamlit app
├── Dockerfile           # Container setup instructions
├── requirements.txt     # Python dependencies
├── mlruns/              # (Optional) MLflow experiment logs
└── README.md            # Project documentation


---

## ⚙️ Prerequisites

- Docker Desktop installed (on Windows/Mac/Linux)
- Git (to clone this repo)
- Browser (to view the app)

---

## 🛠️ How to Build and Run with Docker

1. Clone this repository:

```bash
git clone https://github.com/your-folder/mindcare.git
cd mindcare
```

2.Build the Docker image:

```bash
docker build -t mindcare-app .
```

3. Run the container:

```bash
docker run -p 8501:8501 mindcare-app
```

4. Open your browser and visit:

```bash
http://localhost:8501
```

---

🧪 Optional: MLflow Integration
This project optionally integrates MLflow to track your model training metrics and store the trained model artifacts.
To view MLflow dashboard:

Start MLflow UI (outside the Docker container):

```bash
mlflow ui
```

Then open: http://127.0.0.1:5000

--- 

📧 Contact
Built with ❤️ by Gaurav Kohli

👉 GitHub: https://github.com/Gaurav-04-06/MindCare.git
