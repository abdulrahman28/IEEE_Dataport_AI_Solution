import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
import perf

# Start timer
start_time = time.time()

# Load dataset
fn = 'CVD_balanced.csv'
df = pd.read_csv(fn)
print(df.shape)

# Define features and target
X = df.drop(columns=[
    'TenYearCHD', 'exng', 'caa', 'ldl_cholestrol',
    'hdl_cholestrol', 'Triglyceride', 'CPK_MB_Percentage'
])
y = df['TenYearCHD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'X_Training size: {np.shape(X_train)}')
print(f'X_Testing size: {np.shape(X_test)}')


print(f'y_Training size: {np.shape(y_train)}')
print(f'y_Testing size: {np.shape(y_test)}')

# Define pipeline
pipeline = Pipeline([
   # ('imputer', SimpleImputer(strategy='mean')),
   # ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'classifier__max_depth': [None, 5, 10, 15, 20, 25, 30],
}

# Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=kf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Test set evaluation
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print()
perf.all_metrics(y_test, y_pred_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=["No CHD", "CHD"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No CHD", "CHD"])
plt.figure(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, values_format='d')
#plt.title("Confusion Matrix", fontdict={'fontname': 'Times New Roman', 'size': 14})
plt.xlabel('Predicted Label', fontdict={'fontname': 'Times New Roman', 'size': 12})
plt.ylabel('True Label', fontdict={'fontname': 'Times New Roman', 'size': 12})
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.grid(False)
plt.show()

# Save the model (optional)
# joblib.dump(best_model, 'best_rf_model.joblib')

# Cross-validation results
cv_results = pd.DataFrame(grid_search.cv_results_)

# Print hyperparameter tuning results
print("\nBest Hyperparameters:", grid_search.best_params_)
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
print(f"Test Accuracy on Unseen Data: {test_accuracy:.4f}")

# Execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

# Plot CV accuracy
plt.figure(figsize=(10, 6))
for max_depth in param_grid['classifier__max_depth']:
    subset = cv_results[cv_results['param_classifier__max_depth'] == max_depth]
    plt.plot(subset['param_classifier__n_estimators'], subset['mean_test_score'],
             label=f'max_depth={max_depth}', marker='o')

plt.xlabel('Number of Trees (n_estimators)', fontdict={'fontname': 'Times New Roman', 'size': 12})
plt.ylabel('Cross-validation Accuracy', fontdict={'fontname': 'Times New Roman', 'size': 12})
plt.legend(title='Max Depth', loc='lower right')
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# ROC Curve (binary classification)
y_proba_test = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate', fontdict={'fontname': 'Times New Roman', 'size': 12})
plt.ylabel('True Positive Rate', fontdict={'fontname': 'Times New Roman', 'size': 12})
#plt.title('ROC Curve (Binary Classification)', fontdict={'fontname': 'Times New Roman', 'size': 14})
plt.legend(loc='lower right')
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
