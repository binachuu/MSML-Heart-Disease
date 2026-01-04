import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

EXPERIMENT_NAME = "training-model"
DATA_PATH = "heart_disease_clean.csv"
TARGET_COL = "num"

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.autolog()

    df = pd.read_csv(DATA_PATH)
    df[TARGET_COL] = (df[TARGET_COL] > 0).astype(int)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SATU MODEL SAJA (Basic)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()