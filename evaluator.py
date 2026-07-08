import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


def evaluate_models(trained_models, X_test, y_test, task):
    """
    Evaluate all trained models and return:
    - Comparison DataFrame
    - Best model
    - Best score
    """

    results = []

    best_model = None
    best_model_name = ""
    best_score = -999999

    for name, model in trained_models.items():

        predictions = model.predict(X_test)

        # -------------------------
        # Classification Metrics
        # -------------------------
        if task == "classification":

            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results.append({
                "Model": name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4)
            })

            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_model_name = name

        # -------------------------
        # Regression Metrics
        # -------------------------
        else:

            r2 = r2_score(y_test, predictions)

            mae = mean_absolute_error(
                y_test,
                predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

            results.append({
                "Model": name,
                "R² Score": round(r2, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4)
            })

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name

    results_df = pd.DataFrame(results)

    return (
        results_df,
        best_model,
        best_model_name,
        best_score
    )