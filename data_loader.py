import pandas as pd


def load_data(uploaded_file):
    """
    Load CSV dataset uploaded from Streamlit.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def dataset_summary(df):
    """
    Returns basic dataset information.
    """
    summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }

    return summary


def clean_dataset(df):
    """
    Basic cleaning.
    Removes duplicate rows.
    Removes useless columns.
    """

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove columns containing only one unique value
    constant_cols = [
        col for col in df.columns
        if df[col].nunique() <= 1
    ]

    if constant_cols:
        df = df.drop(columns=constant_cols)

    # Remove common ID columns
    id_columns = [
        col for col in df.columns
        if "id" in col.lower()
    ]

    if id_columns:
        df = df.drop(columns=id_columns)

    return df