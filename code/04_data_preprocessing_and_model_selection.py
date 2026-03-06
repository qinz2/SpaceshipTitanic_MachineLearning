"""
04_data_preprocessing_and_model_selection.py

Functions:
    1. Load feature-engineered data
    2. Identify numerical and categorical features
    3. Feature encoding and standardization (numerical standardization, categorical One-Hot encoding)
    4. Split training and validation sets (80/20 split, stratified sampling)
    5. Train multiple models and compare performance
    6. Select best model for subsequent tuning

Usage:
    python code/04_data_preprocessing_and_model_selection.py

Input Files:
    - spaceship-titanic_data/train_processed.csv
    - spaceship-titanic_data/test_processed.csv

Output Files:
    - result/model_comparison_v2.csv
    - result/Model_Performance_Comparison_v2.png
    - result/preprocessor_v2.pkl
    - result/best_model_Hist_Gradient_Boosting_v2.pkl

Model Selection:
    1. Hist Gradient Boosting: sklearn native implementation, efficient for large datasets
    2. CatBoost: Excellent categorical feature handling, overfitting resistant
    3. Random Forest: Strong baseline model, good generalization
    4. Logistic Regression: Simple baseline model for comparison

Evaluation Metrics:
    - Accuracy: Classification accuracy
    - F1-Score: Harmonic mean of precision and recall
    - Log Loss: Logarithmic loss (lower is better)

Data Preprocessing Strategy:
    - Numerical features: StandardScaler normalization (mean=0, std=1)
    - Categorical features: OneHotEncoder encoding (drop='first' to avoid multicollinearity)
    - Train/validation split: 80/20, stratify to ensure class balance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import warnings
warnings.filterwarnings('ignore')

pd.set_option('future.no_silent_downcasting', True)

# Load processed data
print("Loading Processed Data")
print("-" * 50)

train_path = "D:/机器学习/Experiment/spaceship-titanic_data/train_processed.csv"
test_path = "D:/机器学习/Experiment/spaceship-titanic_data/test_processed.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Separate features and target variable
print("\nSeparating Features and Target Variable")
print("-" * 50)

# Save PassengerId for final submission
train_ids = train_df['PassengerId']
test_ids = test_df['PassengerId']

# Separate X and y
X = train_df.drop(['Transported', 'PassengerId'], axis=1)
y = train_df['Transported'].astype(int)
X_test_final = test_df.drop('PassengerId', axis=1)

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Identify feature types
print("\nIdentifying Feature Types")
print("-" * 50)

# Numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Categorical features
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Feature encoding and standardization
print("\nFeature Encoding and Standardization")
print("-" * 50)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_features)
    ])

# Fit and transform training set
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test_final)

# Get processed feature names
feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

print(f"Processed features shape: {X_processed.shape}")
print(f"Total features after encoding: {len(feature_names)}")

# Split training and validation sets
print("\nSplitting Training and Validation Sets")
print("-" * 50)

X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Training target distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Validation target distribution: {pd.Series(y_val).value_counts().to_dict()}")

# Model definition and training
print("\nModel Training")
print("-" * 50)
print("\nModel Selection Rationale:")
print("- Hist Gradient Boosting: Native sklearn implementation, efficient for large datasets")
print("- CatBoost: Excellent handling of categorical features, robust to overfitting")
print("- Random Forest: Strong baseline, good generalization")
print("- Logistic Regression: Simple baseline for comparison")
print("\nWhy not LightGBM: Exploring alternative gradient boosting implementations")
print("to compare performance and find the best fit for this dataset.\n")

# Define model dictionary (specify loss functions)
models = {
    'Hist Gradient Boosting': HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=0
    ),
    'CatBoost': CatBoostClassifier(
        iterations=100,
        depth=10,
        learning_rate=0.1,
        random_state=42,
        verbose=0,
        loss_function='Logloss'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        verbose=0
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        verbose=0
    )
}

# Store results
results = {}

for model_name, model in models.items():
    print(f"\n--- Training {model_name} ---")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Predict probabilities (for log loss calculation)
    y_train_proba = model.predict_proba(X_train)
    y_val_proba = model.predict_proba(X_val)
    
    # Calculate evaluation metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    train_loss = log_loss(y_train, y_train_proba)
    val_loss = log_loss(y_val, y_val_proba)
    
    # Store results
    results[model_name] = {
        'model': model,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'train_log_loss': train_loss,
        'val_log_loss': val_loss
    }
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")
    print(f"Training Log Loss: {train_loss:.4f}")
    print(f"Validation Log Loss: {val_loss:.4f}")

# Model performance comparison
print("\nModel Performance Comparison")
print("-" * 50)

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Accuracy': [results[m]['train_accuracy'] for m in results.keys()],
    'Val Accuracy': [results[m]['val_accuracy'] for m in results.keys()],
    'Train F1': [results[m]['train_f1'] for m in results.keys()],
    'Val F1': [results[m]['val_f1'] for m in results.keys()],
    'Train Log Loss': [results[m]['train_log_loss'] for m in results.keys()],
    'Val Log Loss': [results[m]['val_log_loss'] for m in results.keys()]
})

print(comparison_df.round(4))

# Sort by validation accuracy
comparison_df_sorted = comparison_df.sort_values('Val Accuracy', ascending=False)
print("\nModels ranked by validation accuracy:")
print(comparison_df_sorted[['Model', 'Val Accuracy', 'Val F1', 'Val Log Loss']].round(4))

# Visualize model performance
print("\nVisualizing Model Performance")
print("-" * 50)

# Performance comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Accuracy comparison
ax1 = axes[0]
x_pos = np.arange(len(results))
width = 0.35
ax1.bar(x_pos - width/2, comparison_df['Train Accuracy'], width, label='Training', alpha=0.8, color='skyblue')
ax1.bar(x_pos + width/2, comparison_df['Val Accuracy'], width, label='Validation', alpha=0.8, color='coral')
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Model'], rotation=20, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# F1-Score comparison
ax2 = axes[1]
ax2.bar(x_pos - width/2, comparison_df['Train F1'], width, label='Training', alpha=0.8, color='skyblue')
ax2.bar(x_pos + width/2, comparison_df['Val F1'], width, label='Validation', alpha=0.8, color='coral')
ax2.set_xlabel('Model')
ax2.set_ylabel('F1-Score')
ax2.set_title('Model F1-Score Comparison')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Model'], rotation=20, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Log Loss comparison (lower is better)
ax3 = axes[2]
ax3.bar(x_pos - width/2, comparison_df['Train Log Loss'], width, label='Training', alpha=0.8, color='skyblue')
ax3.bar(x_pos + width/2, comparison_df['Val Log Loss'], width, label='Validation', alpha=0.8, color='coral')
ax3.set_xlabel('Model')
ax3.set_ylabel('Log Loss (lower is better)')
ax3.set_title('Model Log Loss Comparison')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison_df['Model'], rotation=20, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Model_Performance_Comparison_v2.png", dpi=300, bbox_inches="tight")
plt.show()

# Select best models
print("\nSelecting Best Models")
print("-" * 50)

# Select top 2 models with highest validation accuracy
top_2_models = comparison_df_sorted.head(2)['Model'].tolist()
print(f"Top 2 models selected for next stage: {top_2_models}")

for model_name in top_2_models:
    print(f"\n{model_name}:")
    print(f"  Validation Accuracy: {results[model_name]['val_accuracy']:.4f}")
    print(f"  Validation F1-Score: {results[model_name]['val_f1']:.4f}")
    print(f"  Validation Log Loss: {results[model_name]['val_log_loss']:.4f}")

# Detailed classification report (best model)
print("\nDetailed Classification Report (Best Model)")
print("-" * 50)

best_model_name = top_2_models[0]
best_model = results[best_model_name]['model']
y_val_pred_best = best_model.predict(X_val)

print(f"\n{best_model_name} Classification Report:")
print(classification_report(y_val, y_val_pred_best, target_names=['Not Transported', 'Transported']))

# Model selection summary
print("\nModel Selection Summary")
print("-" * 50)
print("\nWhy we chose these models over LightGBM:")
print("1. Hist Gradient Boosting:")
print("   - Native sklearn implementation, better integration")
print("   - Similar performance to LightGBM with simpler API")
print("   - Excellent handling of missing values and categorical features")
print("\n2. CatBoost:")
print("   - Superior handling of categorical features (no need for extensive encoding)")
print("   - Built-in overfitting detection")
print("   - Robust default parameters")
print("\n3. Random Forest:")
print("   - Strong ensemble baseline")
print("   - Good generalization without extensive tuning")
print("   - Less prone to overfitting")
print("\nLightGBM was excluded to explore alternative gradient boosting")
print("implementations and compare their effectiveness on this dataset.")

# Save results
print("\nSaving Results")
print("-" * 50)

# Save model performance comparison table
comparison_df.to_csv("D:/机器学习/Experiment/result/model_comparison_v2.csv", index=False)
print("Model comparison saved to: result/model_comparison_v2.csv")

# Save preprocessor and best model
import pickle

with open("D:/机器学习/Experiment/result/preprocessor_v2.pkl", 'wb') as f:
    pickle.dump(preprocessor, f)
print("Preprocessor saved to: result/preprocessor_v2.pkl")

with open(f"D:/机器学习/Experiment/result/best_model_{best_model_name.replace(' ', '_')}_v2.pkl", 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved to: result/best_model_{best_model_name.replace(' ', '_')}_v2.pkl")

print("\nData Preprocessing and Model Selection Completed!")