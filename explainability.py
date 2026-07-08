import pandas as pd
import matplotlib.pyplot as plt

# Optional SHAP support
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def get_feature_importance(model, feature_names):
    """
    Returns feature importance for tree-based models.
    """

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        return importance_df

    return None


def plot_feature_importance(importance_df, top_n=10):
    """
    Plot Top N Important Features.
    """

    if importance_df is None:
        return None

    top_features = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    ax.set_xlabel("Importance")

    ax.set_ylabel("Feature")

    ax.set_title("Top Feature Importance")

    ax.invert_yaxis()

    return fig


def generate_shap_values(model, X_train):
    """
    Generate SHAP values if SHAP is installed.
    """

    if not SHAP_AVAILABLE:
        return None, None

    try:

        explainer = shap.Explainer(model)

        shap_values = explainer(X_train)

        return explainer, shap_values

    except Exception:

        return None, None


def plot_shap_summary(shap_values, X_train):
    """
    SHAP summary plot.
    """

    if shap_values is None:
        return None

    shap.summary_plot(
        shap_values,
        X_train,
        show=False
    )

    fig = plt.gcf()

    return fig