"""
01_data_exploration_and_preprocessing.py

Functions:
    1. Load training and test datasets
    2. Analyze data types and missing values
    3. Explore target variable distribution
    4. Analyze numerical and categorical features
    5. Generate correlation heatmap
    6. Save visualization charts to result folder

Usage:
    python code/01_data_exploration_and_preprocessing.py

Output Files:
    - result/Target_Variable_Distribution_Pie.png
    - result/Numerical_Features_Boxplot.png
    - result/Numerical_Features_Histogram.png
    - result/Categorical_Features_Transported_Ratio.png
    - result/Correlation_Heatmap.png
"""

# Core data processing libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Step 1: Load Data")
print("-" * 50)

# Absolute paths
train_df = pd.read_csv("D:/机器学习/Experiment/spaceship-titanic_data/train.csv")
test_df = pd.read_csv("D:/机器学习/Experiment/spaceship-titanic_data/test.csv")

# Check data shape
print(f"Training set shape: {train_df.shape} (rows × columns)")
print(f"Test set shape: {test_df.shape} (rows × columns)")

# Display first 5 rows of training set
print("\nFirst 5 rows of training set:")
print(train_df.head())

# Data type and missing value analysis
print("\nStep 2: Data Type and Missing Value Analysis")
print("-" * 50)

# Check data types
print("Data types of training set:")
print(train_df.dtypes)

# Calculate missing value ratios (%)
missing_train = (train_df.isnull().sum() / len(train_df)) * 100
missing_test = (test_df.isnull().sum() / len(test_df)) * 100

# Organize missing value results
missing_df = pd.DataFrame({
    "Training Set Missing Ratio (%)": missing_train.round(2),
    "Test Set Missing Ratio (%)": missing_test.round(2)
}).sort_values("Training Set Missing Ratio (%)", ascending=False)

print("\nMissing value ratio of features (only features with missing values):")
print(missing_df[missing_df["Training Set Missing Ratio (%)"] > 0])

# Target variable distribution analysis
print("\nStep 3: Target Variable Distribution Analysis")
print("-" * 50)

# Count and ratio of target variable
trans_count = train_df["Transported"].value_counts()
trans_ratio = train_df["Transported"].value_counts(normalize=True) * 100

# Print results
trans_stats = pd.DataFrame({
    "Count": trans_count,
    "Ratio (%)": trans_ratio.round(2)
})
print("Distribution of Transported:")
print(trans_stats)

# Create pie chart and save to result folder
plt.figure(figsize=(8, 6))
labels = ["Transported (True)", "Not Transported (False)"]
sizes = trans_count.values
colors = ["skyblue", "lightcoral"]
plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Target Variable: Transported")
plt.axis("equal")
plt.savefig("D:/机器学习/Experiment/result/Target_Variable_Distribution_Pie.png", dpi=300, bbox_inches="tight")
plt.show()

# Numerical feature analysis
print("\nStep 4: Numerical Feature Analysis")
print("-" * 50)

# Select numerical features
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical features: {numeric_cols}")

# Calculate descriptive statistics
numeric_stats = train_df[numeric_cols].describe().round(2)
print("\nDescriptive statistics of numerical features:")
print(numeric_stats)

# Create boxplots and save
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x="Transported", y=col, data=train_df, palette="Set2")
    plt.title(f"Relationship between {col} and Transported")
    plt.xlabel("Transported")
plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Numerical_Features_Boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# Create histograms and save
plt.figure(figsize=(15, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=train_df, x=col, hue="Transported", kde=True, palette="Set2", bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Numerical_Features_Histogram.png", dpi=300, bbox_inches="tight")
plt.show()

# Categorical feature analysis
print("\nStep 5: Categorical Feature Analysis")
print("-" * 50)

# Select categorical features
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
print(f"Categorical features: {cat_cols}")

# Count category distribution for each categorical feature
for col in cat_cols:
    print(f"\n{col} category distribution:")
    cat_count = train_df[col].value_counts()
    cat_ratio = train_df[col].value_counts(normalize=True) * 100
    cat_stats = pd.DataFrame({
        "Count": cat_count,
        "Ratio (%)": cat_ratio.round(2)
    })
    print(cat_stats)

# Create bar charts and save
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols):
    plt.subplot(2, 2, i+1)
    trans_rate = train_df.groupby(col)["Transported"].mean() * 100
    trans_rate.plot(kind="bar", color="skyblue")
    plt.title(f"Transported Ratio by {col}")
    plt.ylabel("Transported Ratio (%)")
    plt.ylim(0, 100)
    # Display values on bars
    for j, v in enumerate(trans_rate):
        plt.text(j, v+1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Categorical_Features_Transported_Ratio.png", dpi=300, bbox_inches="tight")
plt.show()

# Correlation heatmap
print("\nStep 6: Correlation Analysis of Numerical Features")
print("-" * 50)

# Convert target variable to 0/1
train_corr = train_df.copy()
train_corr["Transported"] = train_corr["Transported"].astype(int)

# Calculate correlation matrix
corr_matrix = train_corr[numeric_cols + ["Transported"]].corr()

# Create heatmap and save
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features and Target Variable")
plt.savefig("D:/机器学习/Experiment/result/Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nData exploration completed! Plots have been saved to the result folder.")