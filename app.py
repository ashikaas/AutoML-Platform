import streamlit as st
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from data_loader import load_data, clean_dataset
from preprocessing import preprocess_data
from model_trainer import get_models, train_models
from evaluator import evaluate_models
from explainability import (
    get_feature_importance,
    plot_feature_importance
)

# -------------------------------------------------
# Page Config
# -------------------------------------------------

st.set_page_config(
    page_title="AutoML Platform",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AutoML Platform")

st.markdown("""
Automatically preprocess datasets, detect the machine learning task,
train multiple models, compare performance and download the best model.
""")

# -------------------------------------------------
# Upload Dataset
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file:

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    df = load_data(uploaded_file)
    df = clean_dataset(df)

    # ---------------------------------------------
    # Dataset Preview
    # ---------------------------------------------

    st.subheader("📊 Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------------------
    # Dataset Overview
    # ---------------------------------------------

    st.subheader("📈 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Features", df.shape[1]-1)
    c4.metric("Missing Values", df.isnull().sum().sum())

    with st.expander("View Missing Values"):
        st.dataframe(df.isnull().sum())

    # ---------------------------------------------
    # Correlation Heatmap
    # ---------------------------------------------

    st.subheader("📊 Correlation Heatmap")

    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] > 1:

        fig, ax = plt.subplots(figsize=(8,6))

        corr = numeric.corr()

        im = ax.imshow(corr, aspect="auto")

        plt.colorbar(im)

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)

        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        st.pyplot(fig)

    # ---------------------------------------------
    # Recommended Target
    # ---------------------------------------------

    recommended = None

    for col in df.columns:

        if col.lower() in [
            "target",
            "label",
            "survived",
            "loan_status",
            "churn"
        ]:

            recommended = col
            break

    if recommended:

        st.success(f"🎯 Recommended Target: {recommended}")

    # ---------------------------------------------
    # Target Selection
    # ---------------------------------------------

    target = st.selectbox(
        "Select Target Column",
        df.columns,
        index=list(df.columns).index(recommended)
        if recommended else 0
    )

    st.info(f"🎯 Selected Target: {target}")

    # ---------------------------------------------
    # Run AutoML
    # ---------------------------------------------

    if st.button("🚀 Run AutoML"):

        with st.spinner("⚙️ Cleaning data and training models..."):

            X_train, X_test, y_train, y_test, task, scaler = preprocess_data(
                df,
                target
            )

            models = get_models(task)

            trained_models = train_models(
                models,
                X_train,
                y_train
            )

            results, best_model, best_model_name, best_score = evaluate_models(
                trained_models,
                X_test,
                y_test,
                task
            )

        # -------------------------------------------------
        # Training Complete
        # -------------------------------------------------

        st.success("✅ Training Completed Successfully!")

        # -------------------------------------------------
        # Problem Type
        # -------------------------------------------------

        st.subheader("🧠 Detected Problem Type")

        if task == "classification":
            st.success("🟢 Classification")
            metric_name = "Accuracy"

        else:
            st.success("🔵 Regression")
            metric_name = "R² Score"

        # -------------------------------------------------
        # Model Comparison
        # -------------------------------------------------

        st.subheader("📋 Model Comparison")

        st.dataframe(
            results,
            use_container_width=True
        )

        # -------------------------------------------------
        # Performance Chart
        # -------------------------------------------------

        st.subheader("📈 Model Performance")

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(
            results["Model"],
            results[metric_name]
        )

        plt.xticks(rotation=20)

        ax.set_ylabel(metric_name)

        ax.set_xlabel("Models")

        ax.set_title("Model Performance Comparison")

        st.pyplot(fig)

        # -------------------------------------------------
        # Best Model
        # -------------------------------------------------

        st.subheader("🏆 Best Model")

        col1, col2 = st.columns(2)

        col1.metric(
            "Model",
            best_model_name
        )

        col2.metric(
            metric_name,
            round(best_score,4)
        )

        # -------------------------------------------------
        # Feature Importance
        # -------------------------------------------------

        importance = get_feature_importance(
            best_model,
            X_train.columns
        )

        if importance is not None:

            st.subheader("⭐ Feature Importance")

            fig = plot_feature_importance(
                importance
            )

            st.pyplot(fig)

            st.dataframe(
                importance.head(10),
                use_container_width=True
            )

        # -------------------------------------------------
        # Model Visualization
        # -------------------------------------------------

        if task == "classification":

            st.subheader("📊 Confusion Matrix")

            predictions = best_model.predict(X_test)

            fig, ax = plt.subplots(figsize=(5,5))

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                cmap="Blues",
                ax=ax
            )

            st.pyplot(fig)

        else:

            st.subheader("📈 Actual vs Predicted")

            predictions = best_model.predict(X_test)

            fig, ax = plt.subplots(figsize=(6,6))

            ax.scatter(
                y_test,
                predictions,
                alpha=0.7
            )

            ax.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "r--"
            )

            ax.set_xlabel("Actual")

            ax.set_ylabel("Predicted")

            ax.set_title("Actual vs Predicted")

            st.pyplot(fig)

        # -------------------------------------------------
        # Downloads
        # -------------------------------------------------

        st.subheader("💾 Downloads")

        joblib.dump(best_model, "best_model.pkl")

        with open("best_model.pkl", "rb") as f:

            st.download_button(
                "⬇ Download Trained Model (.pkl)",
                data=f,
                file_name="best_model.pkl",
                mime="application/octet-stream"
            )

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Model Results (.csv)",
            data=csv,
            file_name="model_results.csv",
            mime="text/csv"
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")

st.markdown("""
## 🚀 AutoML Platform

Automatically preprocess datasets, detect the machine learning task,
train multiple ML models, compare their performance, and download
the best performing model.

### ✨ Features

- 📂 Upload any CSV dataset
- 🤖 Automatic Classification / Regression Detection
- 🧹 Automatic Data Preprocessing
- 📊 Multiple ML Algorithms
- 🏆 Best Model Selection
- ⭐ Feature Importance
- 📈 Performance Visualization
- 💾 Model Download

### 🛠 Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib

Developed as an AI & Machine Learning Portfolio Project.
""")