import os
import pandas as pd
import numpy as np
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from scipy import stats

def create_baseline_plots():
    # Configuration
    depths = [1, 2, 3]
    random_seed = 42
    top_n_features = 10
    subset_traces_html = 50  # Number of traces for Plotly HTML
    
    tasks = [
        {
            "name": "regression",
            "data_path": "/workspaces/test_output/regression/california_housing.csv",
            "output_dir": "/workspaces/test_output/regression/new_code/",
            "model_class": cb.CatBoostRegressor,
            "params": {
                "loss_function": "Poisson", 
                "random_seed": random_seed,
                "interpolation_enabled": True,
                "interpolation_type": "Sigmoid",
                "interpolation_span_mode": "Relative",
                "interpolation_span_per_float_feature": {"AveBedrms" : 0.05, "AveOccup" : 0.2 ,
                                                         "AveRooms" : 0.16, "HouseAge" : 0.25,
                                                         "Latitude" : 1/34, "Longitude" : 1/120,
                                                         "MedInc" : 0.25, "Population" : 0.25},

                "monotone_constraints": {"AveBedrms": 1, "AveOccup": -1, "HouseAge": 1,
                                          "Latitude": -1, "Longitude": -1, "MedInc": 1,
                                          "Population": 1}
            }
        },
        {
            "name": "classification",
            "data_path": "/workspaces/test_output/classification/breast_cancer.csv",
            "output_dir": "/workspaces/test_output/classification/new_code/",
            "model_class": cb.CatBoostClassifier,
            "params": {
                "loss_function": "Logloss", 
                "random_seed": random_seed,
                "interpolation_enabled": True,
                "interpolation_type": "Sigmoid",
                "interpolation_span_mode": "Relative",
                "interpolation_span_per_float_feature": {"area error" : 0.25, "mean concave points" : 0.5,
                                                         "mean texture" : 0.2, "perimeter error" : 0.3,
                                                         "radius error" : 1/3, "worst area" : 0.5,
                                                         "worst concave points" : 0.5, "worst concavity" : 1,
                                                         "worst perimeter" : 0.5, "worst radius" : 0.25,
                                                         "worst texture" : 0.3, "mean fractal dimension" : 0.025/0.65},

                "monotone_constraints": {"mean concave points" : -1, "mean fractal dimension": 1,
                                       "perimeter error" : -1,
                                       "radius error" : -1, "worst area" : -1,
                                       "worst concave points" : -1, "worst concavity" : -1,
                                       "worst perimeter" : -1, "worst radius" : -1,
                                       "worst texture" : -1, "mean concavity" : -1}
            }
        }
    ]

    for task in tasks:
        print(f"\nProcessing {task['name']}...")
        os.makedirs(task["output_dir"], exist_ok=True)
        
        # Load data
        df = pd.read_csv(task["data_path"])
        X = df.drop(columns=["target"])
        y = df["target"]

        # 70:30 split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=random_seed
        )

        for depth in depths:
            print(f"  Training model with depth {depth}...")
            params = task["params"].copy()
            params["interpolation_span_per_float_feature"] = {
                X.columns.get_loc(feature_name): span
                for feature_name, span in params["interpolation_span_per_float_feature"].items()
            }
            params["max_depth"] = depth
            params["n_estimators"] = 1000 # Increase estimators to allow early stopping to work
            
            model = task["model_class"](**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=False)

            # Get top N features by importance
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n_features]
            top_features = X.columns[indices]

            for feature in top_features:
                print(f"    Generating plots for feature: {feature}")
                
                # Calculate PDP and ICE
                # For classification, we want probability (index 1 if binary)
                response_method = "auto"
                if task["name"] == "classification":
                    # For binary classification, partial_dependence by default returns 
                    # the decision function or probability for the positive class if response_method is 'auto'.
                    pass

                pd_results = partial_dependence(
                    model, X, [feature], kind="both", grid_resolution=50
                )
                
                grid_values = pd_results["grid_values"][0]
                # For regression, n_outputs=1. For binary classification, n_outputs=1 or 2.
                # sklearn 1.8.0 might return (n_outputs, ...)
                
                # Let's handle both 1D and 2D/3D shapes to be safe
                ice_raw = pd_results["individual"]
                pdp_raw = pd_results["average"]
                
                if ice_raw.ndim == 3: # (n_outputs, n_samples, n_grid_points)
                    ice_preds = ice_raw[0]
                else: # (n_samples, n_grid_points)
                    ice_preds = ice_raw
                    
                if pdp_raw.ndim == 2: # (n_outputs, n_grid_points)
                    pdp_preds = pdp_raw[0]
                else: # (n_grid_points,)
                    pdp_preds = pdp_raw

                # Calculate rescaling factor based on mode or median of feature
                feature_values = X[feature]
                # Use median as a robust central tendency
                ref_val = feature_values.median()
                
                # Find index in grid closest to ref_val
                idx_ref = np.argmin(np.abs(grid_values - ref_val))
                
                # Rescale ICE: divide each trace by its value at ref_val
                # Add small epsilon to avoid division by zero if necessary, 
                # but predictions are usually non-zero for these models.
                ice_rescaled = []
                for i in range(ice_preds.shape[0]):
                    val_at_ref = ice_preds[i, idx_ref]
                    if abs(val_at_ref) < 1e-9:
                         ice_rescaled.append(ice_preds[i] / 1e-9)
                    else:
                        ice_rescaled.append(ice_preds[i] / val_at_ref)
                ice_rescaled = np.array(ice_rescaled)
                
                # Rescale PDP similarly for consistency in the plot
                pdp_val_at_ref = pdp_preds[idx_ref]
                if abs(pdp_val_at_ref) < 1e-9:
                    pdp_rescaled = pdp_preds / 1e-9
                else:
                    pdp_rescaled = pdp_preds / pdp_val_at_ref

                # --- Matplotlib Plot (PNG) ---
                plt.figure(figsize=(10, 6))
                # Plot ICE (sub-sample for PNG to keep it clean)
                n_ice_plot = min(100, ice_rescaled.shape[0])
                idx_ice_plot = np.random.choice(ice_rescaled.shape[0], n_ice_plot, replace=False)
                for i in idx_ice_plot:
                    plt.plot(grid_values, ice_rescaled[i], color="gray", alpha=0.1, linewidth=0.5)
                
                # Plot PDP
                plt.plot(grid_values, pdp_rescaled, color="blue", linewidth=2, label="PDP (Rescaled)")
                plt.title(f"Depth {depth} - Feature: {feature} (Rescaled at median)")
                plt.xlabel(feature)
                plt.ylabel("Relative Prediction Change")
                plt.legend()
                
                png_path = os.path.join(task["output_dir"], f"depth_{depth}_{feature}_pdp_ice.png")
                plt.savefig(png_path)
                plt.close()


if __name__ == "__main__":
    create_baseline_plots()
