import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def detect_task(y):
    """
    Automatically detect whether the problem is
    Classification or Regression.
    """

    if y.dtype == "object":
        return "classification"

    if y.nunique() <= 10:
        return "classification"

    return "regression"


def preprocess_data(df, target_column):
    """
    Complete preprocessing pipeline.
    """

    df = df.copy()

    # -----------------------------
    # Remove duplicate rows
    # -----------------------------
    df = df.drop_duplicates()

    # -----------------------------
    # Fill Missing Values
    # -----------------------------
    for col in df.columns:

        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")

        else:
            df[col] = df[col].fillna(df[col].median())

    # -----------------------------
    # Separate Features and Target
    # -----------------------------
    X = df.drop(columns=[target_column])

    y = df[target_column]

    # -----------------------------
    # Encode Target
    # -----------------------------
    if y.dtype == "object":

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y)

    # -----------------------------
    # Encode Categorical Features
    # -----------------------------
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns

    if len(categorical_columns) > 0:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True
        )

    # -----------------------------
    # Scale Numeric Features
    # -----------------------------
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if detect_task(pd.Series(y)) == "classification" else None
    )

    task = detect_task(pd.Series(y))

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        task,
        scaler
    )