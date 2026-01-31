
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
data = pd.read_csv("data/churn.csv")

X = data.drop("Churn", axis=1)
y = data["Churn"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. MLflow experiment
mlflow.set_experiment("Customer_Churn_Experiment")

with mlflow.start_run():
    # 4. Model
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    model.fit(X_train, y_train)

    # 5. Predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 6. Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", accuracy)
