import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 👇 Connect to MLflow UI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MindCare")

def log_model(model, X_test, y_test, model_name="MindCareModel"):
    try:
        with mlflow.start_run():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_param("model", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            print(f"✅ Logged to MLflow with accuracy: {acc:.4f}")
    except Exception as e:
        print("❌ MLflow logging failed:", e)

# Train and log
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

log_model(model, X_test, y_test)
