import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("HousingData.csv")

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()

X_train_processed = scaler.fit_transform(X_train_imputed)
X_test_processed = scaler.transform(X_test_imputed)

print("\nPreprocessing completed.")
print("Missing values handled and features scaled.")

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

linear_model = LinearRegression()

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

linear_model.fit(X_train_processed, y_train)
random_forest_model.fit(X_train_processed, y_train)

print("\nModels trained successfully.")

linear_pred = linear_model.predict(X_test_processed)
random_forest_pred = random_forest_model.predict(X_test_processed)

print("Predictions generated successfully.")

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)

rf_mae = mean_absolute_error(y_test, random_forest_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, random_forest_pred))
rf_r2 = r2_score(y_test, random_forest_pred)

print("\nModel Evaluation:")
print("----------------------------")

print("Linear Regression:")
print("MAE :", linear_mae)
print("RMSE:", linear_rmse)
print("R²  :", linear_r2)

print("\nRandom Forest:")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)

joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(random_forest_model, "random_forest_model.pkl")

print("\nFinal Random Forest model saved successfully.")

plt.figure(figsize=(8, 6))

plt.scatter(y_test, random_forest_pred)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()

print("\nFinal Model: Random Forest")
print("Reason: It achieved lower MAE and RMSE and higher R² than Linear Regression.")