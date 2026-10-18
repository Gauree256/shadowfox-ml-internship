
import pandas as pd
import joblib

model = joblib.load("02-loan-approval/model.pkl")

applicant = pd.DataFrame([{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500.0,
    "LoanAmount": 120.0,
    "Loan_Amount_Term": 360.0,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}])

prediction = model.predict(applicant)[0]

if prediction == 1:
    print("Predicted Loan Status: Approved (Y)")
else:
    print("Predicted Loan Status: Rejected (N)")

probabilities = model.predict_proba(applicant)[0]

print("\nModel-estimated class probabilities:")
for label, probability in zip(
    model.classes_,
    probabilities
):
    status = "Approved" if label == 1 else "Rejected"
    print(f"{status}: {probability:.2%}")
