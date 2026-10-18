
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix
)

df = pd.read_csv("02-loan-approval/loan_prediction.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

X = df.drop(columns=["Loan_ID", "Loan_Status"])
y = df["Loan_Status"].map({"N": 0, "Y": 1})

if y.isnull().any():
    raise ValueError("Unexpected or missing values in Loan_Status.")

numeric_columns = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude="number"
).columns.tolist()

print("\nNumerical columns:", numeric_columns)
print("Categorical columns:", categorical_columns)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
print("\nModel training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Rejected", "Approved"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=["Rejected", "Approved"],
    cmap="viridis"
)

plt.title("Loan Approval Prediction")
plt.tight_layout()
plt.savefig("02-loan-approval/confusion_matrix.png", dpi=150)
plt.show()

joblib.dump(model, "02-loan-approval/model.pkl")

print("\nModel saved to 02-loan-approval/model.pkl")
print("Confusion matrix saved to 02-loan-approval/confusion_matrix.png")
