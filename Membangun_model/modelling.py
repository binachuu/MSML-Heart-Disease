import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =====================
# CONFIG
# =====================
EXPERIMENT_NAME = "training-model"
DATA_PATH = "heart_disease_clean.csv" 
TARGET_COL = "num"


# =====================
# MAIN
# =====================
def main():
    # set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # load data
    df = pd.read_csv(DATA_PATH)

    # binary classification (0 = sehat, 1 = sakit)
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =====================
    # LOGISTIC REGRESSION
    # =====================
    with mlflow.start_run(run_name="LogisticRegression"):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)

        y_pred_lr = lr.predict(X_test)
        acc_lr = accuracy_score(y_test, y_pred_lr)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc_lr)

        mlflow.sklearn.log_model(
            lr,
            artifact_path="model",
            registered_model_name="HeartDiseaseModel_LR"
        )

        print(f"Accuracy LR: {acc_lr}")

    # =====================
    # RANDOM FOREST
    # =====================
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc_rf)

        mlflow.sklearn.log_model(
            rf,
            artifact_path="model",
            registered_model_name="HeartDiseaseModel_RF"
        )

        print(f"Accuracy RF: {acc_rf}")


if __name__ == "__main__":
    main()