# ---------------------------------------
# Model Training Script - Tourism Project
# ---------------------------------------

import pandas as pd
import os
import joblib
import mlflow

# Data Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model Training & Evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Hugging Face Hub
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# Configuration
# -----------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "tourism-package-predict-rf"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "muthukumar22/Tourism-Package-Predict"
HF_MODEL_REPO = "muthukumar22/tourism-package-mod"

# -----------------------------
# Setup MLflow
# -----------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------
# Load Data from Hugging Face
# -----------------------------
print("Loading data from Hugging Face...")
try:
    Xtrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtrain.csv")
    Xtest  = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/Xtest.csv")
    ytrain = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytrain.csv").values.ravel()
    ytest  = pd.read_csv(f"hf://datasets/{HF_DATASET_REPO}/ytest.csv").values.ravel()
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# -----------------------------
# Feature Definitions
# -----------------------------
numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar", 
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", 
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus", 
    "Designation", "ProductPitched"
]

# -----------------------------
# Build Pipeline (with Imputation)
# -----------------------------
# Random Forest requires no missing values, so we add Imputers.

numeric_transformer = make_pipeline(
    SimpleImputer(strategy="median"), 
    StandardScaler()
)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features)
)

# -----------------------------
# Random Forest Model
# -----------------------------
# class_weight='balanced' handles the class imbalance automatically
rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Combine into pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

# -----------------------------
# Hyperparameter Grid
# -----------------------------
# Note: The prefix 'randomforestclassifier__' comes from the class name in the pipeline
param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [None, 10, 20],
    "randomforestclassifier__min_samples_split": [2, 5],
    "randomforestclassifier__min_samples_leaf": [1, 2],
    "randomforestclassifier__max_features": ["sqrt", "log2"]
}

# -----------------------------
# Training Execution
# -----------------------------
print("Starting MLflow run...")

with mlflow.start_run() as run:
    
    # Run GridSearch
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="recall", # optimizing for Recall
        n_jobs=-1
    )
    
    grid_search.fit(Xtrain, ytrain)

    # -----------------------------
    # Log Child Runs
    # -----------------------------
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_score", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_score", results["std_test_score"][i])

    # -----------------------------
    # Log Best Parameters & Model
    # -----------------------------
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    
    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test  = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    }
    
    mlflow.log_metrics(metrics)
    print("Metrics logged.")

    # -----------------------------
    # Save & Upload Artifacts
    # -----------------------------
    model_filename = "best_tourism_model_rf.joblib"
    joblib.dump(best_model, model_filename)
    
    # Log to MLflow
    mlflow.log_artifact(model_filename)
    
    # Upload to Hugging Face
    if HF_TOKEN:
        print("Uploading model to Hugging Face...")
        api = HfApi(token=HF_TOKEN)
        
        # Create repo if it doesn't exist
        try:
            api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        except RepositoryNotFoundError:
            create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)
            print(f"Created new model repository: {HF_MODEL_REPO}")

        api.upload_file(
            path_or_fileobj=model_filename,
            path_in_repo=model_filename,
            repo_id=HF_MODEL_REPO,
            repo_type="model"
        )
        print("Upload successful.")
    else:
        print("HF_TOKEN not found. Skipping Hugging Face upload.")

print("Training complete.")
