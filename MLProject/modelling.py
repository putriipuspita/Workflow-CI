# =====================================================
# IMPORT LIBRARY
# =====================================================
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# DAGSHUB CONFIG
# =====================================================
# dagshub
dagshub.init(
    repo_owner="putriipuspita",
    repo_name="eksperimen-sml",
    mlflow=True
)

mlflow.set_experiment("random-forest-basic")

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("customer_churn_preprocessing.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAINING MODEL
# =====================================================
with mlflow.start_run():

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # =================================================
    # MANUAL LOGGING
    # =================================================
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)

    # save model
    mlflow.sklearn.log_model(model, "model")

    # artifact tambahan (biar aman rubric advanced)
    mlflow.log_text(str(model.get_params()), "params.txt")
    mlflow.log_dict(
        {"accuracy": accuracy},
        "metrics.json"
    )

    print("Accuracy:", accuracy)