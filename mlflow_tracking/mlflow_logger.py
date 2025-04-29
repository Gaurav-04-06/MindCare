import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for plots
import matplotlib.pyplot as plt
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MindCare-Depression-Prediction")

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("final_depression_dataset.csv")

# Split dataset
students_df = df[df['Working Professional or Student'] == "Student"].copy()
professionals_df = df[df['Working Professional or Student'] == "Working Professional"].copy()

# Drop unnecessary columns
students_df = students_df.drop(columns=["Work Pressure", "Profession", "Job Satisfaction"])
professionals_df = professionals_df.drop(columns=["Academic Pressure", "CGPA", "Study Satisfaction"])

# Drop missing values
students_df = students_df.dropna()
professionals_df = professionals_df.dropna()

# Encoding
binary_columns_students = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
non_binary_columns_students = ['City', 'Dietary Habits', 'Sleep Duration', 'Degree']
binary_columns_professionals = ['Gender', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
non_binary_columns_professionals = ['City', 'Dietary Habits', 'Sleep Duration', 'Degree', 'Profession']

le = LabelEncoder()
for col in binary_columns_students:
    students_df[col] = le.fit_transform(students_df[col])
students_df = pd.get_dummies(students_df, columns=non_binary_columns_students)

for col in binary_columns_professionals:
    professionals_df[col] = le.fit_transform(professionals_df[col])
professionals_df = pd.get_dummies(professionals_df, columns=non_binary_columns_professionals)

# Prepare data
X_students = students_df.drop(columns=["Depression", "Working Professional or Student", "Name"])
y_students = le.fit_transform(students_df["Depression"])
X_professionals = professionals_df.drop(columns=["Depression", "Working Professional or Student", "Name"])
y_professionals = le.fit_transform(professionals_df["Depression"])

# Save feature order
print("📦 Saving feature order...")
os.makedirs("artifacts", exist_ok=True)
with open("artifacts/feature_order_students.pkl", "wb") as f:
    pickle.dump(X_students.columns.tolist(), f)
with open("artifacts/feature_order_professionals.pkl", "wb") as f:
    pickle.dump(X_professionals.columns.tolist(), f)

# Train-test split
X_train_stu, X_test_stu, y_train_stu, y_test_stu = train_test_split(X_students, y_students, test_size=0.2, random_state=42)
X_train_pro, X_test_pro, y_train_pro, y_test_pro = train_test_split(X_professionals, y_professionals, test_size=0.3, random_state=42)

# Grid search parameters
param_grid = {'n_estimators': [30], 'max_depth': [5], 'min_samples_split': [2]}

# Train Student model
print("🧠 Training student model...")
grid_stu = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=5)
grid_stu.fit(X_train_stu, y_train_stu)
y_pred_stu = grid_stu.predict(X_test_stu)

print("📊 Logging student model to MLflow...")
with mlflow.start_run(run_name="Student Model"):
    mlflow.set_tag("type", "student")
    mlflow.log_params(grid_stu.best_params_)
    acc = accuracy_score(y_test_stu, y_pred_stu)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(grid_stu.best_estimator_, "student_model")
    mlflow.log_artifact("artifacts/feature_order_students.pkl")
    report = classification_report(y_test_stu, y_pred_stu, output_dict=True)
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                mlflow.log_metric(f"student_{label}_{metric}", value)
    fi = grid_stu.best_estimator_.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(X_train_stu.columns, fi)
    plt.title("Student Feature Importances")
    plt.tight_layout()
    plt.savefig("artifacts/student_feature_importance.png")
    mlflow.log_artifact("artifacts/student_feature_importance.png")

# Train Professional model
print("🧠 Training professional model...")
grid_pro = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=5)
grid_pro.fit(X_train_pro, y_train_pro)
y_pred_pro = grid_pro.predict(X_test_pro)

print("📊 Logging professional model to MLflow...")
with mlflow.start_run(run_name="Professional Model"):
    mlflow.set_tag("type", "professional")
    mlflow.log_params(grid_pro.best_params_)
    acc = accuracy_score(y_test_pro, y_pred_pro)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(grid_pro.best_estimator_, "professional_model")
    mlflow.log_artifact("artifacts/feature_order_professionals.pkl")
    report = classification_report(y_test_pro, y_pred_pro, output_dict=True)
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                mlflow.log_metric(f"professional_{label}_{metric}", value)
    fi = grid_pro.best_estimator_.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(X_train_pro.columns, fi)
    plt.title("Professional Feature Importances")
    plt.tight_layout()
    plt.savefig("artifacts/professional_feature_importance.png")
    mlflow.log_artifact("artifacts/professional_feature_importance.png")

print("✅ All done! Check MLflow UI at http://127.0.0.1:5000")
