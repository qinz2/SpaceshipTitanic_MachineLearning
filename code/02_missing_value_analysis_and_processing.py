"""
02_missing_value_analysis_and_processing.py

Functions:
    1. Analyze missing value distribution in training and test sets
    2. Compare missing value ratios between datasets
    3. Analyze missing value correlations
    4. Handle missing values in numerical and categorical features
    5. Generate missing value analysis visualization charts

Usage:
    python code/02_missing_value_analysis_and_processing.py

Output Files:
    - result/Missing_Value_Comparison_Processing.png
    - result/Missing_Value_Correlation_Heatmap_Processing.png
    - result/Numerical_Features_Correlation_Heatmap_Processing.png

Missing Value Handling Strategy:
    - Age: Median imputation (avoid extreme value impact)
    - RoomService/FoodCourt/ShoppingMall/Spa/VRDeck: Fill with 0 (no consumption)
    - HomePlanet/CryoSleep/Destination/VIP: Mode imputation (most common category)
    - Cabin: Fill with "Missing" (preserve missing information)
    - Name: Fill with "Unknown"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Solve pandas type conversion warning
pd.set_option('future.no_silent_downcasting', True)

# Load data (absolute paths)
train_path = "D:/机器学习/Experiment/spaceship-titanic_data/train.csv"
test_path = "D:/机器学习/Experiment/spaceship-titanic_data/test.csv"

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Data loaded successfully")
except FileNotFoundError:
    print("Data file not found, please check the path")
    exit()

# Missing value analysis
print("\nMissing Value Analysis")
print("-" * 50)

# Calculate missing value ratios (%)
missing_train = (train_df.isnull().sum() / len(train_df)) * 100
missing_test = (test_df.isnull().sum() / len(test_df)) * 100

# Organize missing value results
missing_df = pd.DataFrame({
    "Training Set Missing Ratio (%)": missing_train.round(2),
    "Test Set Missing Ratio (%)": missing_test.round(2)
}).sort_values("Training Set Missing Ratio (%)", ascending=False)

# Show only features with missing values
missing_features = missing_df[missing_df["Training Set Missing Ratio (%)"] > 0]
print("Missing value ratio of each feature:")
print(missing_features)

print("\nTop 5 Features with Higher Missing Ratios:")
print(missing_features.head())

# Training set vs test set missing value comparison chart
plt.figure(figsize=(12, 6))
train_missing = train_df.drop("Transported", axis=1).isnull().mean().sort_values(ascending=True)
test_missing = test_df.isnull().mean().reindex(train_missing.index).sort_values(ascending=True)

x_pos = np.arange(len(train_missing))
width = 0.35
plt.barh(x_pos - width/2, train_missing * 100, width, label="Training Set", color="skyblue")
plt.barh(x_pos + width/2, test_missing * 100, width, label="Test Set", color="lightcoral")

plt.xlabel("Missing Ratio (%)")
plt.ylabel("Features")
plt.title("Missing Value Ratio Comparison: Training Set vs Test Set")
plt.legend()
plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Missing_Value_Comparison_Processing.png", dpi=300, bbox_inches="tight")
plt.show()

# Missing value correlation analysis
print("\nMissing Value Correlation Analysis")
print("-" * 50)

missing_corr = train_df.isnull().corr().round(2)
print("Correlation matrix of missing values (1 = high co-missing probability):")
print(missing_corr)

# Plot missing value correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(missing_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Missing Values")
plt.savefig("D:/机器学习/Experiment/result/Missing_Value_Correlation_Heatmap_Processing.png", dpi=300, bbox_inches="tight")
plt.show()

# Descriptive statistics of numerical features
print("\nDescriptive Statistics of Numerical Features")
print("-" * 50)

numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_stats = train_df[numeric_cols].describe().round(2)
print(numeric_stats)

print("\nExtreme Values of Numerical Features (Outlier Reference):")
for col in numeric_cols:
    max_val = train_df[col].max()
    q95 = train_df[col].quantile(0.95)
    print(f"{col}: Max value = {max_val}, 95th percentile = {q95.round(2)} (Values exceeding this are considered outliers)")

# Correlation analysis
print("\nCorrelation Between Numerical Features and Target Variable")
print("-" * 50)

train_corr = train_df.copy()
train_corr["Transported"] = train_corr["Transported"].astype(int)
corr_matrix = train_corr[numeric_cols + ["Transported"]].corr()
corr_with_target = corr_matrix["Transported"].sort_values(ascending=False).round(2)
print(corr_with_target)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features and Target Variable")
plt.savefig("D:/机器学习/Experiment/result/Numerical_Features_Correlation_Heatmap_Processing.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nLow Correlation Features (|r| < 0.1, Consider Deleting Later):")
low_corr_features = [col for col in numeric_cols if abs(corr_matrix["Transported"][col]) < 0.1]
if low_corr_features:
    print(low_corr_features)
else:
    print("No low correlation features")

# Handle missing values in numerical features
print("\nHandling Missing Values of Numerical Features")
print("-" * 50)

train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
consume_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in consume_cols:
    train_df[col] = train_df[col].fillna(0)

print("Missing values after filling numerical features:")
print(train_df[numeric_cols].isnull().sum())

# Handle missing values in categorical features
print("\nHandling Missing Values of Categorical Features")
print("-" * 50)

cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
for col in cat_cols:
    train_df[col] = train_df[col].fillna(train_df[col].mode()[0])
train_df["Cabin"] = train_df["Cabin"].fillna("Missing")
train_df["Name"] = train_df["Name"].fillna("Unknown")

check_cols = cat_cols + ["Cabin", "Name"]
print("Missing values after filling categorical features:")
print(train_df[check_cols].isnull().sum())

# Final validation
print("\nFinal Validation: Total Missing Values in Training Set")
print("-" * 50)

total_missing = train_df.isnull().sum().sum()
if total_missing == 0:
    print(f"All missing values have been handled. Total missing values: {total_missing}")
else:
    print(f"There are still {total_missing} missing values, please check the processing logic.")

print("\nMissing Value Analysis and Processing Completed!")