from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Optional XGBoost
try:
    from xgboost import (
        XGBClassifier,
        XGBRegressor
    )
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models(task):
    """
    Returns ML models depending on task type.
    """

    if task == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                random_state=42
            ),
            "KNN": KNeighborsClassifier(),
            "Support Vector Machine": SVC()
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                eval_metric="logloss",
                random_state=42
            )

        return models

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                random_state=42
            )
        }

        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(
                random_state=42
            )

        return models


def train_models(models, X_train, y_train):
    """
    Train every model.
    """

    trained_models = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models