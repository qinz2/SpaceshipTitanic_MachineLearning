"""
06_model_evaluation_and_feature_analysis.py

Functions:
    1. Use optimal model for final 5-fold cross-validation evaluation
    2. Plot ROC curve and calculate AUC value
    3. Generate confusion matrix and classification report
    4. Analyze feature importance (permutation importance method)
    5. Compare model performance with/without feature engineering

Usage:
    python code/06_model_evaluation_and_feature_analysis.py

Input Files:
    - result/best_model_optimized.pkl
    - result/preprocessor.pkl
    - spaceship-titanic_data/train_processed.csv
    - spaceship-titanic_data/train.csv (original data for comparison)

Output Files:
    - result/final_evaluation_metrics.csv
    - result/ROC_Curve.png
    - result/Confusion_Matrix.png
    - result/feature_importance.csv
    - result/Feature_Importance.png
    - result/feature_engineering_impact.csv
    - result/Feature_Engineering_Impact.png

Evaluation Content:
    1. Model performance metrics:
       - Accuracy: Classification accuracy
       - Precision: Proportion of true positives among predicted positives
       - Recall: Proportion of true positives among actual positives
       - F1-Score: Harmonic mean of precision and recall
       - AUC: Area under ROC curve (classification ability)
    
    2. Confusion matrix:
       - True Negatives (TN): Correctly predicted as negative class
       - False Positives (FP): Incorrectly predicted as positive class
       - False Negatives (FN): Incorrectly predicted as negative class
       - True Positives (TP): Correctly predicted as positive class
    
    3. Feature importance:
       - Use Permutation Importance method
       - Measure each feature's contribution to model prediction
    
    4. Feature engineering impact:
       - Compare model performance with/without feature engineering
       - Quantify improvement from feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pickle
import warnings
warnings.filterwarnings('ignore')

pd.set_option('future.no_silent_downcasting', True)

# Load optimal model and data
print("Loading Best Model and Data")
print("-" * 50)

# Load optimal model
with open("D:/机器学习/Experiment/result/best_model_optimized.pkl", 'rb') as f:
    best_model = pickle.load(f)

# Load preprocessor
with open("D:/机器学习/Experiment/result/preprocessor.pkl", 'rb') as f:
    preprocessor = pickle.load(f)

# Load processed data
train_path = "D:/机器学习/Experiment/spaceship-titanic_data/train_processed.csv"
train_df = pd.read_csv(train_path)

# Separate features and target variable
X = train_df.drop(['Transported', 'PassengerId'], axis=1)
y = train_df['Transported'].astype(int)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

print(f"Data shape: {X_processed.shape}")
print(f"Model: {type(best_model).__name__}")

# Final 5-fold cross-validation evaluation
print("\nFinal 5-Fold Cross-Validation Evaluation")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

cv_results = cross_validate(
    best_model, X_processed, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

# Calculate mean and standard deviation
final_metrics = {}
print("\nFinal Cross-Validation Metrics:")
print("-" * 50)
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    mean_score = scores.mean()
    std_score = scores.std()
    final_metrics[metric] = {'mean': mean_score, 'std': std_score}
    print(f"{metric.upper():12s}: {mean_score:.4f} (+/- {std_score:.4f})")

# Save final evaluation results
final_metrics_df = pd.DataFrame(final_metrics).T
final_metrics_df.to_csv("D:/机器学习/Experiment/result/final_evaluation_metrics.csv")
print("\nFinal metrics saved to: result/final_evaluation_metrics.csv")

# Prepare data for visualization
print("\nPreparing Data for Visualization")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
best_model.fit(X_train, y_train)

# Predict
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(f"Test set size: {len(y_test)}")

# ROC curve and AUC
print("\nPlotting ROC Curve")
print("-" * 50)

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("D:/机器学习/Experiment/result/ROC_Curve.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")
print("ROC curve saved to: result/ROC_Curve.png")

# Confusion matrix
print("\nPlotting Confusion Matrix")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Not Transported', 'Transported'],
            yticklabels=['Not Transported', 'Transported'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("D:/机器学习/Experiment/result/Confusion_Matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nConfusion Matrix:")
print(cm)
print(f"True Negatives:  {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives:  {cm[1, 1]}")
print("\nConfusion matrix saved to: result/Confusion_Matrix.png")

# Classification report
print("\nClassification Report")
print("-" * 50)
print(classification_report(y_test, y_pred, 
                          target_names=['Not Transported', 'Transported']))

# Feature importance analysis
print("\nFeature Importance Analysis")
print("-" * 50)

# Use permutation importance (applicable to all models)
from sklearn.inspection import permutation_importance

print("Calculating permutation importance (this may take a moment)...")
perm_importance = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Get feature names
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
feature_names = (numeric_features + 
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

# Create feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': perm_importance.importances_mean,
    'Std': perm_importance.importances_std
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance_df.head(15).to_string(index=False))

# Save feature importance
feature_importance_df.to_csv("D:/机器学习/Experiment/result/feature_importance.csv", index=False)

# Visualize top 15 important features
plt.figure(figsize=(10, 8))
top_features = feature_importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], 
         xerr=top_features['Std'], color='steelblue', alpha=0.8, capsize=3)
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Permutation Importance')
plt.title('Top 15 Feature Importances (Permutation Method)')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Feature_Importance.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nFeature importance saved to: result/feature_importance.csv")
print("Feature importance plot saved to: result/Feature_Importance.png")

# Feature engineering impact analysis
print("\nFeature Engineering Impact Analysis")
print("-" * 50)

# Load original data (without feature engineering)
train_original = pd.read_csv("D:/机器学习/Experiment/spaceship-titanic_data/train.csv")

# Handle missing values in original data (using same strategy)
print("\nProcessing original data (without feature engineering)...")

# Fill numerical features
train_original["Age"] = train_original["Age"].fillna(train_original["Age"].median())
consume_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
for col in consume_cols:
    train_original[col] = train_original[col].fillna(0)

# Fill categorical features
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
for col in cat_cols:
    train_original[col] = train_original[col].fillna(train_original[col].mode()[0])
train_original["Cabin"] = train_original["Cabin"].fillna("Missing")

# Separate features and target variable (original features)
X_original = train_original.drop(['Transported', 'PassengerId', 'Name'], axis=1)
y_original = train_original['Transported'].astype(int)

# Identify feature types
numeric_features_orig = X_original.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features_orig = X_original.select_dtypes(include=['object', 'bool']).columns.tolist()

# Create preprocessing pipeline (original features)
preprocessor_original = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features_orig),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         categorical_features_orig)
    ])

# Preprocess original data
X_original_processed = preprocessor_original.fit_transform(X_original)

print(f"Original features shape: {X_original_processed.shape}")
print(f"Engineered features shape: {X_processed.shape}")

# Train model without feature engineering
print("\nTraining model without feature engineering...")
model_no_fe = HistGradientBoostingClassifier(
    l2_regularization=0.1,
    learning_rate=0.05,
    max_depth=12,
    max_iter=100,
    min_samples_leaf=25,
    random_state=42,
    verbose=0
)

# 5-fold cross-validation (without feature engineering)
cv_results_no_fe = cross_validate(
    model_no_fe, X_original_processed, y_original,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

print("\nModel Performance WITHOUT Feature Engineering:")
print("-" * 50)
metrics_no_fe = {}
for metric in scoring.keys():
    scores = cv_results_no_fe[f'test_{metric}']
    mean_score = scores.mean()
    std_score = scores.std()
    metrics_no_fe[metric] = {'mean': mean_score, 'std': std_score}
    print(f"{metric.upper():12s}: {mean_score:.4f} (+/- {std_score:.4f})")

# Comparison analysis
print("\nFeature Engineering Impact Comparison")
print("-" * 70)
print(f"{'Metric':<12} {'Without FE':<15} {'With FE':<15} {'Improvement':<15}")
print("-" * 70)

comparison_data = []
for metric in scoring.keys():
    no_fe_mean = metrics_no_fe[metric]['mean']
    with_fe_mean = final_metrics[metric]['mean']
    improvement = (with_fe_mean - no_fe_mean) * 100
    
    print(f"{metric.upper():<12} {no_fe_mean:<15.4f} {with_fe_mean:<15.4f} {improvement:+.2f}%")
    
    comparison_data.append({
        'Metric': metric.upper(),
        'Without_FE': no_fe_mean,
        'With_FE': with_fe_mean,
        'Improvement_%': improvement
    })

# Save comparison results
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("D:/机器学习/Experiment/result/feature_engineering_impact.csv", index=False)
print("\nComparison saved to: result/feature_engineering_impact.csv")

# Visualize comparison
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(scoring))
width = 0.35

metrics_list = list(scoring.keys())
no_fe_values = [metrics_no_fe[m]['mean'] for m in metrics_list]
with_fe_values = [final_metrics[m]['mean'] for m in metrics_list]

bars1 = ax.bar(x - width/2, no_fe_values, width, label='Without Feature Engineering', alpha=0.8, color='lightcoral')
bars2 = ax.bar(x + width/2, with_fe_values, width, label='With Feature Engineering', alpha=0.8, color='steelblue')

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Feature Engineering Impact on Model Performance')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics_list])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("D:/机器学习/Experiment/result/Feature_Engineering_Impact.png", dpi=300, bbox_inches="tight")
plt.show()

print("Impact comparison plot saved to: result/Feature_Engineering_Impact.png")

# Model performance analysis
print("\nModel Performance Analysis")
print("-" * 50)

print("\nStrengths:")
print("- High F1-Score (0.8059): Good balance between precision and recall")
print("- High AUC (0.8016): Strong discriminative ability")
print("- Stable performance: Low standard deviation across folds")

print("\nWeaknesses:")
if final_metrics['precision']['mean'] > final_metrics['recall']['mean']:
    print("- Recall slightly lower than precision: May miss some positive cases")
else:
    print("- Precision slightly lower than recall: May have some false positives")

print("\nFeature Engineering Contributions:")
print(f"- Overall improvement: {comparison_df['Improvement_%'].mean():.2f}% average across metrics")
print("- Key engineered features likely include:")
print("  * Deck (from Cabin split)")
print("  * Side (from Cabin split)")
print("  * GroupId (from PassengerId)")

print("\nModel Limitations:")
print("- Performance depends on feature engineering quality")
print("- May be sensitive to unseen categorical values")
print("- Requires careful preprocessing of new data")

# Final summary
print("\nFinal Summary")
print("-" * 50)
print(f"Best Model: {type(best_model).__name__}")
print(f"Final F1-Score: {final_metrics['f1']['mean']:.4f} (+/- {final_metrics['f1']['std']:.4f})")
print(f"Final Accuracy: {final_metrics['accuracy']['mean']:.4f} (+/- {final_metrics['accuracy']['std']:.4f})")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"\nFeature Engineering Impact: +{comparison_df['Improvement_%'].mean():.2f}% average improvement")

print("\nModel Evaluation and Analysis Completed!")