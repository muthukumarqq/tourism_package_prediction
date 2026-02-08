# ---------------------------------------
# Data Preparation Script - Tourism Project 
# ---------------------------------------

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from huggingface_hub import HfApi

# -----------------------------
# Configuration
# -----------------------------
CONFIG = {
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "REPO_ID": "muthukumar22/Tourism-Package-Predict",
    "DATASET_URL": "hf://datasets/muthukumar22/Tourism-Package-Predict/tourism.csv",
    "TARGET_COL": "ProdTaken",
    "DROP_COLS": ["CustomerID"],
    "TEST_SIZE": 0.2,
    "RANDOM_STATE": 42,
    "OUTPUT_DIR": "processed_data"
}

def load_data(url):
    """Loads dataset from URL."""
    try:
        df = pd.read_csv(url)
        print(f"Dataset loaded: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def create_preprocessor(X):
    """Creates a Scikit-Learn pipeline for preprocessing."""
    # Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    # Pipeline for Categorical: Handle Missing -> Convert to Number
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Pipeline for Numerical: Handle Missing
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        verbose_feature_names_out=False
    )
    
    return preprocessor

def main():
    # 1. Load Data
    df = load_data(CONFIG["DATASET_URL"])

    # 2. Drop unnecessary columns
    df.drop(columns=CONFIG["DROP_COLS"], inplace=True, errors='ignore')

    # 3. Split Features and Target
    X = df.drop(columns=[CONFIG["TARGET_COL"]])
    y = df[CONFIG["TARGET_COL"]]

    # 4. Train-Test Split (CRITICAL: Do this BEFORE encoding to prevent leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["TEST_SIZE"], 
        random_state=CONFIG["RANDOM_STATE"], 
        stratify=y
    )

    # 5. Build and Fit Preprocessor
    print("Fitting preprocessor on training data...")
    preprocessor = create_preprocessor(X_train)
    
    # Fit on Train, Transform Train and Test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert back to DataFrame to keep headers
    feat_names = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=feat_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feat_names)

    # 6. Save Local Files
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    paths = {
        "Xtrain.csv": X_train_df,
        "Xtest.csv": X_test_df,
        "ytrain.csv": y_train,
        "ytest.csv": y_test
    }

    file_list = []
    for filename, data in paths.items():
        path = os.path.join(CONFIG["OUTPUT_DIR"], filename)
        data.to_csv(path, index=False)
        file_list.append(path)

    # 7. Save the Preprocessor (Vital for inference later)
    preprocessor_path = os.path.join(CONFIG["OUTPUT_DIR"], "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    file_list.append(preprocessor_path)

    print(f"Files saved locally in {CONFIG['OUTPUT_DIR']}/")

    # 8. Upload to Hugging Face
    if CONFIG["HF_TOKEN"]:
        print("Uploading to Hugging Face...")
        api = HfApi(token=CONFIG["HF_TOKEN"])
        
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_name,
                repo_id=CONFIG["REPO_ID"],
                repo_type="dataset",
            )
        print("Upload complete.")
    else:
        print("Warning: HF_TOKEN not found. Skipping upload.")

if __name__ == "__main__":
    main()
