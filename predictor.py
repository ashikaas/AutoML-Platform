import joblib
import pandas as pd


def save_model(model, filename="best_model.pkl"):
    """
    Save the trained model.
    """
    joblib.dump(model, filename)


def load_model(filename="best_model.pkl"):
    """
    Load the saved model.
    """
    return joblib.load(filename)


def predict(model, input_df):
    """
    Predict using the trained model.

    Parameters
    ----------
    model : trained sklearn model
    input_df : pandas DataFrame

    Returns
    -------
    prediction
    """

    prediction = model.predict(input_df)

    return prediction


def predict_single(model, feature_names, values):
    """
    Predict a single record.

    Example:
    feature_names = ["Age","Fare","Pclass"]
    values = [25, 50, 2]
    """

    df = pd.DataFrame([values], columns=feature_names)

    prediction = model.predict(df)

    return prediction[0]