"""
07_generate_test_predictions.py

Functions:
    1. Load optimal model and preprocessor
    2. Retrain model on full training set (utilize all data)
    3. Predict on test set
    4. Generate Kaggle submission file (submission.csv)
    5. Validate submission file format
    6. Generate prediction statistics and detailed prediction results

Usage:
    python code/07_generate_test_predictions.py

Input Files:
    - result/best_model_optimized.pkl
    - result/preprocessor.pkl
    - spaceship-titanic_data/train_processed.csv
    - spaceship-titanic_data/test_processed.csv

Output Files:
    - result/submission.csv (Kaggle submission file)
    - result/detailed_predictions.csv (detailed results with prediction probabilities)

Submission File Format:
    - Columns: PassengerId, Transported
    - Transported type: bool (True/False)
    - No missing values
    - PassengerId unique and consistent with test set

Prediction Strategy:
    1. Retrain using full training set (don't waste validation set data)
    2. Transform test set using preprocessor fitted on training set (avoid data leakage)
    3. Generate prediction labels and prediction probabilities
    4. Analyze prediction confidence distribution

Important Notes:
    - Test set preprocessing must use training set statistics (mean, std, categories, etc.)
    - Cannot use test set information to fit preprocessor
    - Ensure training and test sets use same feature engineering pipeline
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

pd.set_option('future.no_silent_downcasting', True)

# Load optimal model and preprocessor
print("Loading Best Model and Preprocessor")
print("-" * 50)

# Load optimal model
with open("D:/机器学习/Experiment/result/best_model_optimized.pkl", 'rb') as f:
    best_model = pickle.load(f)

# Load preprocessor
with open("D:/机器学习/Experiment/result/preprocessor.pkl", 'rb') as f:
    preprocessor = pickle.load(f)

print(f"Model: {type(best_model).__name__}")
print(f"Model parameters: {best_model.get_params()}")

# Load processed training and test sets
print("\nLoading Processed Data")
print("-" * 50)

train_path = "D:/机器学习/Experiment/spaceship-titanic_data/train_processed.csv"
test_path = "D:/机器学习/Experiment/spaceship-titanic_data/test_processed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Save PassengerId
train_ids = train_df['PassengerId']
test_ids = test_df['PassengerId']

# Prepare training data (full training set)
print("\nPreparing Full Training Data")
print("-" * 50)

# Separate features and target variable
X_train_full = train_df.drop(['Transported', 'PassengerId'], axis=1)
y_train_full = train_df['Transported'].astype(int)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train_full)

print(f"Full training set features shape: {X_train_processed.shape}")
print(f"Full training set target shape: {y_train_full.shape}")
print(f"Target distribution: {y_train_full.value_counts().to_dict()}")

# Retrain model on full training set
print("\nRetraining Model on Full Training Set")
print("-" * 50)

# Train using optimal parameters on full training set
best_model.fit(X_train_processed, y_train_full)

print("Model retrained successfully on full training set")
print(f"Training samples used: {len(y_train_full)}")

# Prepare test set data
print("\nPreparing Test Data")
print("-" * 50)

# Separate test set features
X_test = test_df.drop('PassengerId', axis=1)

# Apply preprocessing (using preprocessor fitted on training set)
X_test_processed = preprocessor.transform(X_test)

print(f"Test set features shape: {X_test_processed.shape}")
print("Preprocessing applied using training set statistics (no data leakage)")

# Predict test set
print("\nPredicting Test Set")
print("-" * 50)

# Predict
y_test_pred = best_model.predict(X_test_processed)
y_test_proba = best_model.predict_proba(X_test_processed)

print(f"Predictions generated for {len(y_test_pred)} samples")
print(f"Predicted distribution: {pd.Series(y_test_pred).value_counts().to_dict()}")
print(f"Predicted ratio (Transported): {y_test_pred.mean():.4f}")

# Generate submission file
print("\nGenerating Submission File")
print("-" * 50)

# Create submission DataFrame
submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': y_test_pred.astype(bool)
})

# Save submission file
submission_path = "D:/机器学习/Experiment/result/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file saved to: {submission_path}")
print(f"Submission shape: {submission.shape}")
print("\nFirst 10 predictions:")
print(submission.head(10))

# Validate submission file format
print("\nValidating Submission File")
print("-" * 50)

# Check column names
required_columns = ['PassengerId', 'Transported']
if list(submission.columns) == required_columns:
    print("Column names are correct")
else:
    print("Column names are incorrect")
    print(f"  Expected: {required_columns}")
    print(f"  Got: {list(submission.columns)}")

# Check number of rows
if len(submission) == len(test_df):
    print(f"Number of predictions matches test set ({len(submission)})")
else:
    print(f"Number of predictions mismatch")
    print(f"  Expected: {len(test_df)}")
    print(f"  Got: {len(submission)}")

# Check missing values
if submission.isnull().sum().sum() == 0:
    print("No missing values in submission")
else:
    print("Missing values found in submission")
    print(submission.isnull().sum())

# Check data types
if submission['Transported'].dtype == bool:
    print("Transported column is boolean type")
else:
    print(f"Transported column type is {submission['Transported'].dtype}")

# Check PassengerId uniqueness
if submission['PassengerId'].nunique() == len(submission):
    print("All PassengerIds are unique")
else:
    print("Duplicate PassengerIds found")

print("\nSubmission File Validation Complete")

# Generate prediction statistics
print("\nPrediction Statistics")
print("-" * 50)

# Prediction distribution
pred_counts = pd.Series(y_test_pred).value_counts()
print("\nPrediction Distribution:")
print(f"Not Transported (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(y_test_pred)*100:.2f}%)")
print(f"Transported (1):     {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(y_test_pred)*100:.2f}%)")

# Prediction probability statistics
print("\nPrediction Probability Statistics:")
print(f"Mean probability: {y_test_proba[:, 1].mean():.4f}")
print(f"Std probability:  {y_test_proba[:, 1].std():.4f}")
print(f"Min probability:  {y_test_proba[:, 1].min():.4f}")
print(f"Max probability:  {y_test_proba[:, 1].max():.4f}")

# High confidence predictions
high_confidence = (y_test_proba.max(axis=1) > 0.9).sum()
low_confidence = (y_test_proba.max(axis=1) < 0.6).sum()
print(f"\nHigh confidence predictions (>0.9): {high_confidence} ({high_confidence/len(y_test_pred)*100:.2f}%)")
print(f"Low confidence predictions (<0.6):  {low_confidence} ({low_confidence/len(y_test_pred)*100:.2f}%)")

# Save prediction probabilities (for analysis)
print("\nSaving Prediction Probabilities")
print("-" * 50)

# Create detailed prediction results
detailed_predictions = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': y_test_pred.astype(bool),
    'Probability_Not_Transported': y_test_proba[:, 0],
    'Probability_Transported': y_test_proba[:, 1],
    'Confidence': y_test_proba.max(axis=1)
})

detailed_path = "D:/机器学习/Experiment/result/detailed_predictions.csv"
detailed_predictions.to_csv(detailed_path, index=False)

print(f"Detailed predictions saved to: {detailed_path}")

# Compare train and test prediction distributions
print("\nComparing Train and Test Distributions")
print("-" * 50)

train_transported_ratio = y_train_full.mean()
test_transported_ratio = y_test_pred.mean()

print(f"Training set Transported ratio: {train_transported_ratio:.4f}")
print(f"Test set Transported ratio:     {test_transported_ratio:.4f}")
print(f"Difference: {abs(train_transported_ratio - test_transported_ratio):.4f}")

if abs(train_transported_ratio - test_transported_ratio) < 0.05:
    print("Train and test distributions are similar (good sign)")
else:
    print("Train and test distributions differ significantly")

# Final summary
print("\nFinal Summary")
print("-" * 50)
print(f"Model: {type(best_model).__name__}")
print(f"Training samples: {len(y_train_full)}")
print(f"Test samples: {len(y_test_pred)}")
print(f"Submission file: {submission_path}")
print(f"Detailed predictions: {detailed_path}")
print("\nNext Steps:")
print("1. Review the submission file: result/submission.csv")
print("2. Submit to Kaggle: https://www.kaggle.com/competitions/spaceship-titanic")
print("3. Record the score in the report")

print("\nTest Set Prediction Completed!")