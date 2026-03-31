import os
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer

def download_datasets():
    # Define paths
    regression_dir = "/workspaces/test_output/regression/"
    classification_dir = "/workspaces/test_output/classification/"
    
    # Ensure directories exist (though user mentioned they do, good practice)
    os.makedirs(regression_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)
    
    print("Downloading California Housing dataset (Regression)...")
    cal_housing = fetch_california_housing()
    df_reg = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    df_reg['target'] = cal_housing.target
    df_reg.to_csv(os.path.join(regression_dir, "california_housing.csv"), index=False)
    print(f"Saved to {regression_dir}california_housing.csv")
    
    print("\nDownloading Breast Cancer dataset (Classification)...")
    breast_cancer = load_breast_cancer()
    df_clf = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    df_clf['target'] = breast_cancer.target
    df_clf.to_csv(os.path.join(classification_dir, "breast_cancer.csv"), index=False)
    print(f"Saved to {classification_dir}breast_cancer.csv")

if __name__ == "__main__":
    download_datasets()
