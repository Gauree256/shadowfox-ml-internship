# Loan Approval Prediction using Machine Learning

## 1. Project Overview

This project uses machine learning to predict whether a loan
application is likely to be approved or rejected.

It is developed as part of the ShadowFox Machine Learning
Engineer Internship, Intermediate task.

## 2. Problem Statement

Loan approval depends on several applicant and financial factors.
This project trains a classification model using historical loan
application data to predict the loan status.

- Y: Approved
- N: Rejected

The model is intended for educational purposes and is not a
replacement for real financial or lending decisions.

## 3. Dataset

File: `loan_prediction.csv`

Dataset dimensions: 614 rows and 13 columns.

The dataset contains applicant details, income, loan amount,
loan term, credit history, property area, and loan status.

`Loan_ID` is excluded from model training because it is an
identifier rather than a meaningful applicant feature.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## 5. Data Preprocessing

The training pipeline performs the following operations:

1. Separates the input features from the target.
2. Splits data into training and testing sets.
3. Fills missing numerical values using the median.
4. Fills missing categorical values using the most frequent value.
5. Scales numerical features.
6. Encodes categorical features using one-hot encoding.

Preprocessing is fitted using the training data to reduce data
leakage from the test set.

## 6. Machine Learning Model

Algorithm: Random Forest Classifier

Configuration:
- Number of trees: 200
- Random state: 42
- Class weighting: balanced

## 7. Model Evaluation

The initial test run achieved approximately 80.49% accuracy.

The initial confusion matrix was:

| Actual / Predicted | Rejected | Approved |
|---|---:|---:|
| Rejected | 25 | 13 |
| Approved | 11 | 74 |

The classification report and confusion matrix are printed when
the training script is executed.

These results represent one train/test split and do not guarantee
the same performance on future applications.

## 8. Project Structure

02-loan-approval/
- loan_prediction.csv
- train.py
- predict.py
- model.pkl
- confusion_matrix.png
- README.md
- requirements.txt

## 9. Installation

Run the following command from the repository root:

```bash
python -m pip install -r 02-loan-approval/requirements.txt
```

## 10. Train the Model

Run from the repository root:

```bash
python 02-loan-approval/train.py
```

This trains the model, prints evaluation metrics, saves the
confusion matrix, and writes the fitted pipeline to `model.pkl`.

## 11. Make a Prediction

First, run the training script.

Then run:

```bash
python 02-loan-approval/predict.py
```

The prediction script loads the saved model and predicts the
outcome for a sample loan application.

## 12. Limitations

- The dataset is relatively small.
- Historical patterns may not reflect current lending practices.
- The model may make incorrect predictions.
- Accuracy alone is insufficient for assessing a lending model.
- The model is for educational demonstration, not real loan
  approval decisions.
