"""
05_cross_validation_and_hyperparameter_tuning.py

Functions:
    1. Use 5-fold cross-validation to evaluate model performance (more reliable evaluation)
    2. Grid search hyperparameter tuning for Hist Gradient Boosting
    3. Compare baseline and optimized model performance
    4. Select optimal model and save

Usage:
    python code/05_cross_validation_and_hyperparameter_tuning.py

Input Files:
    - spaceship-titanic_data/train_processed.csv
    - result/preprocessor.pkl

Output Files:
    - result/cv_comparison.csv
    - result/CV_Performance_Comparison.png
    - result/best_model_optimized.pkl
    - result/best_params.csv

Cross-Validation Strategy:
    - 5-fold stratified cross-validation (StratifiedKFold)
    - Ensure consistent class proportions in each fold
    - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC

Hyperparameter Tuning Range:
    - max_iter: [100, 150, 200] - Number of iterations
    - max_depth: [8, 10, 12] - Maximum tree depth
    - learning_rate: [0.05, 0.1, 0.15] - Learning rate
    - min_samples_leaf: [15, 20, 25] - Minimum samples per leaf
    - l2_regularization: [0, 0.1, 0.5] - L2 regularization coefficient

Tuning Method:
    - GridSearchCV: Grid search all parameter combinations
    - Optimization target: F1-Score (balance precision and recall)
    - Parallel computation: n_jobs=-1 (use all CPU cores)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

pd.set_option('future.no_silent_downcasting', True)

# Load preprocessed data
print("Loading Preprocessed Data")
print("-" * 50)

# Load preprocessor
with open("D:/机器学习/Experiment/result/preprocessor.pkl", 'rb') as f:
    preprocessor = pickle.load(f)

# Load original processed data
train_path = "D:/机器学习/Experiment/spaceship-titanic_data/train_processed.csv"
train_df = pd.read_csv(train_path)

# Separate features and target variable
X = train_df.drop(['Transported', 'PassengerId'], axis=1)
y = train_df['Transported'].astype(int)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

print(f"Features shape: {X_processed.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Define evaluation metrics
print("\nDefining Evaluation Metrics")
print("-" * 50)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

print("Evaluation metrics: Accuracy, Precision, Recall, F1-Score, AUC")

# Hist Gradient Boosting - 5-fold cross-validation (baseline)
print("\n5-Fold Cross-Validation: Hist Gradient Boosting (Baseline)")
print("-" * 50)

# Define 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Baseline model
baseline_hist = HistGradientBoostingClassifier(
    max_iter=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)

# Execute 5-fold cross-validation
cv_scores_baseline = cross_validate(
    baseline_hist, X_processed, y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print("\nBaseline Hist Gradient Boosting - Cross-Validation Results:")
baseline_results = {}
for metric in scoring.keys():
    train_scores = cv_scores_baseline[f'train_{metric}']
    test_scores = cv_scores_baseline[f'test_{metric}']
    
    baseline_results[f'{metric}_train_mean'] = train_scores.mean()
    baseline_results[f'{metric}_train_std'] = train_scores.std()
    baseline_results[f'{metric}_test_mean'] = test_scores.mean()
    baseline_results[f'{metric}_test_std'] = test_scores.std()
    
    print(f"{metric.upper()}:")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
    print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")

# CatBoost - Basic evaluation (using train_test_split)
print("\nCatBoost: Basic Evaluation")
print("-" * 50)

# Split dataset for CatBoost evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Train CatBoost
catboost_model = CatBoostClassifier(
    iterations=100,
    depth=10,
    learning_rate=0.1,
    random_state=42,
    verbose=0,
    loss_function='Logloss'
)

catboost_model.fit(X_train, y_train)

# Evaluate
y_pred = catboost_model.predict(X_val)
y_proba = catboost_model.predict_proba(X_val)[:, 1]

catboost_results = {
    'accuracy': accuracy_score(y_val, y_pred),
    'precision': precision_score(y_val, y_pred),
    'recall': recall_score(y_val, y_pred),
    'f1': f1_score(y_val, y_pred),
    'roc_auc': roc_auc_score(y_val, y_proba)
}

print("\nCatBoost Validation Results:")
for metric, value in catboost_results.items():
    print(f"{metric.upper()}: {value:.4f}")

# Hyperparameter tuning - Hist Gradient Boosting
print("\nHyperparameter Tuning: Hist Gradient Boosting")
print("-" * 50)

# Define parameter grid
param_grid_hist = {
    'max_iter': [100, 150, 200],
    'max_depth': [8, 10, 12],
    'learning_rate': [0.05, 0.1, 0.15],
    'min_samples_leaf': [15, 20, 25],
    'l2_regularization': [0, 0.1, 0.5]
}

# Create GridSearchCV object
grid_search_hist = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42, verbose=0),
    param_grid_hist,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print("Starting grid search for Hist Gradient Boosting...")
print(f"Total combinations: {np.prod([len(v) for v in param_grid_hist.values()])}")

# Execute grid search
grid_search_hist.fit(X_processed, y)

print(f"\nBest parameters: {grid_search_hist.best_params_}")
print(f"Best F1-Score: {grid_search_hist.best_score_:.4f}")

# Complete cross-validation with best parameters
best_hist_model = grid_search_hist.best_estimator_
hist_cv_scores = cross_validate(
    best_hist_model, X_processed, y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print("\nOptimized Hist Gradient Boosting - Cross-Validation Results:")
optimized_results = {}
for metric in scoring.keys():
    train_scores = hist_cv_scores[f'train_{metric}']
    test_scores = hist_cv_scores[f'test_{metric}']
    
    optimized_results[f'{metric}_train_mean'] = train_scores.mean()
    optimized_results[f'{metric}_train_std'] = train_scores.std()
    optimized_results[f'{metric}_test_mean'] = test_scores.mean()
    optimized_results[f'{metric}_test_std'] = test_scores.std()
    
    print(f"{metric.upper()}:")
    print(f"  Train: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
    print(f"  Test:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")

# Model performance comparison
print("\nModel Performance Comparison")
print("-" * 50)

# Create comparison table
comparison_data = []

# Hist GB baseline
row1 = {'Model': 'Hist Gradient Boosting (Baseline)'}
for metric in scoring.keys():
    row1[f'{metric}_mean'] = baseline_results[f'{metric}_test_mean']
    row1[f'{metric}_std'] = baseline_results[f'{metric}_test_std']
comparison_data.append(row1)

# Hist GB optimized
row2 = {'Model': 'Hist Gradient Boosting (Optimized)'}
for metric in scoring.keys():
    row2[f'{metric}_mean'] = optimized_results[f'{metric}_test_mean']
    row2[f'{metric}_std'] = optimized_results[f'{metric}_test_std']
comparison_data.append(row2)

# CatBoost
row3 = {'Model': 'CatBoost (Baseline)'}
for metric in scoring.keys():
    row3[f'{metric}_mean'] = catboost_results[metric]
    row3[f'{metric}_std'] = 0.0  # Single evaluation has no standard deviation
comparison_data.append(row3)

comparison_df = pd.DataFrame(comparison_data)
print("\nPerformance Comparison:")
print(comparison_df.to_string(index=False))

# Save comparison results
comparison_df.to_csv("D:/机器学习/Experiment/result/cv_comparison.csv", index=False)
print("\nComparison saved to: result/cv_comparison.csv")

# Visualize cross-validation results
print("\nVisualizing Cross-Validation Results")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_list = list(scoring.keys())

for idx, metric in enumerate(metrics_list):
    ax = axes[idx // 3, idx % 3]
    
    models = comparison_df['Model'].tolist()
    means = comparison_df[f'{metric}_mean'].tolist()
    stds = comparison_df[f'{metric}_std'].tolist()
    
    x_pos = np.arange(len(models))
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
    ax.set_xlabel('Model')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/CV_Performance_Comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Performance improvement analysis
print("\nPerformance Improvement Analysis")
print("-" * 50)

print("\nBaseline vs Optimized (Hist Gradient Boosting):")
for metric in scoring.keys():
    baseline_val = baseline_results[f'{metric}_test_mean']
    optimized_val = optimized_results[f'{metric}_test_mean']
    improvement = (optimized_val - baseline_val) * 100
    
    print(f"{metric.upper()}:")
    print(f"  Baseline:  {baseline_val:.4f}")
    print(f"  Optimized: {optimized_val:.4f}")
    print(f"  Improvement: {improvement:+.2f}%")

# Determine optimal model
print("\nSelecting Best Model")
print("-" * 50)

# Select optimal model based on F1-Score
best_f1_idx = comparison_df['f1_mean'].idxmax()
best_model_name = comparison_df.loc[best_f1_idx, 'Model']
best_f1_score = comparison_df.loc[best_f1_idx, 'f1_mean']

print(f"Best Model: {best_model_name}")
print(f"Best F1-Score: {best_f1_score:.4f}")

# Select optimal model
if 'Optimized' in best_model_name:
    final_model = best_hist_model
    best_params = grid_search_hist.best_params_
else:
    final_model = baseline_hist
    best_params = baseline_hist.get_params()

# Train optimal model on full training set
print(f"\nTraining final model on full dataset...")
final_model.fit(X_processed, y)

# Save optimal model
print("\nSaving Best Model")
print("-" * 50)

# Save optimal model
model_path = "D:/机器学习/Experiment/result/best_model_optimized.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Best model saved to: {model_path}")

# Save best parameters
params_df = pd.DataFrame([best_params])
params_df.to_csv("D:/机器学习/Experiment/result/best_params.csv", index=False)
print(f"Best parameters saved to: result/best_params.csv")

# Final summary
print("\nFinal Summary")
print("-" * 50)
print(f"Best Model: {best_model_name}")
print(f"Best Parameters: {best_params}")
print("\nCross-Validation Performance:")
for metric in scoring.keys():
    mean_val = comparison_df.loc[best_f1_idx, f'{metric}_mean']
    std_val = comparison_df.loc[best_f1_idx, f'{metric}_std']
    print(f"  {metric.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")

print("\nCross-Validation and Model Tuning Completed!")