# =====================================================
# IMPORT LIBRARY
# =====================================================
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# MLFLOW AUTO LOGGING
# =====================================================
mlflow.set_experiment("random-forest-basic")
mlflow.autolog()

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
with mlflow.start_run() as run:

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # write run_id to file (penting untuk CI/CD)
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)