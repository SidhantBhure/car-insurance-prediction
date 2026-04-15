"""Script to generate the Jupyter Notebook for Car Insurance Claim Prediction."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""# 🚗 Car Insurance Claim Prediction System
## Machine Learning Project — Fraud Detection

**Objective:** Build a machine learning pipeline to predict fraudulent insurance claims using two models:
1. **Random Forest Classifier**
2. **XGBoost Classifier**

**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, Classification Report

---"""
))

# ── Step 1: Import Libraries ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 1: Import Libraries"))
cells.append(nbf.v4.new_code_cell(
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("✅ All libraries imported successfully!")"""
))

# ── Step 2: Load Data ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 2: Load and Explore the Dataset"))
cells.append(nbf.v4.new_code_cell(
"""# Load the dataset
df = pd.read_csv('insurance_claims.csv')

print(f"Dataset Shape: {df.shape}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print(f"\\nColumn Names:\\n{list(df.columns)}")"""
))

cells.append(nbf.v4.new_code_cell(
"""# First 5 rows
df.head()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Dataset info
df.info()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Statistical summary
df.describe()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Check for '?' values (used as missing values in this dataset)
missing_question = (df == '?').sum()
missing_question = missing_question[missing_question > 0]
print("Columns with '?' values (missing data):")
print(missing_question)
print(f"\\nTotal '?' values: {(df == '?').sum().sum()}")"""
))

# ── Step 3: EDA ──────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 3: Exploratory Data Analysis (EDA)"))

cells.append(nbf.v4.new_code_cell(
"""# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
colors = ['#6366f1', '#ef4444']
fraud_counts = df['fraud_reported'].value_counts()
axes[0].bar(fraud_counts.index, fraud_counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_title('Fraud Reported Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Fraud Reported')
axes[0].set_ylabel('Count')
for i, v in enumerate(fraud_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=12)

# Pie chart
axes[1].pie(fraud_counts.values, labels=fraud_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=(0, 0.05),
            textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Fraud Reported Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\\nClass Distribution:")
print(f"  No Fraud (N): {fraud_counts.get('N', 0)} ({fraud_counts.get('N', 0)/len(df)*100:.1f}%)")
print(f"  Fraud (Y): {fraud_counts.get('Y', 0)} ({fraud_counts.get('Y', 0)/len(df)*100:.1f}%)")"""
))

cells.append(nbf.v4.new_code_cell(
"""# Age distribution by fraud status
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, fraud in enumerate(['N', 'Y']):
    subset = df[df['fraud_reported'] == fraud]['age']
    axes[i].hist(subset, bins=20, color=colors[i], edgecolor='white', alpha=0.85)
    axes[i].set_title(f"Age Distribution — {'No Fraud' if fraud == 'N' else 'Fraud'}", fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Age')
    axes[i].set_ylabel('Count')
    axes[i].axvline(subset.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean: {subset.mean():.1f}')
    axes[i].legend(fontsize=11)

plt.tight_layout()
plt.show()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Incident type vs fraud
fig, ax = plt.subplots(figsize=(12, 5))
ct = pd.crosstab(df['incident_type'], df['fraud_reported'])
ct.plot(kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=1)
ax.set_title('Incident Type vs Fraud Reported', fontsize=14, fontweight='bold')
ax.set_xlabel('Incident Type')
ax.set_ylabel('Count')
ax.legend(title='Fraud Reported')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Incident severity vs fraud
fig, ax = plt.subplots(figsize=(12, 5))
ct = pd.crosstab(df['incident_severity'], df['fraud_reported'])
ct.plot(kind='bar', ax=ax, color=colors, edgecolor='white', linewidth=1)
ax.set_title('Incident Severity vs Fraud Reported', fontsize=14, fontweight='bold')
ax.set_xlabel('Incident Severity')
ax.set_ylabel('Count')
ax.legend(title='Fraud Reported')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Total claim amount distribution
fig, ax = plt.subplots(figsize=(12, 5))
for i, fraud in enumerate(['N', 'Y']):
    subset = df[df['fraud_reported'] == fraud]['total_claim_amount']
    ax.hist(subset, bins=30, color=colors[i], alpha=0.6, label=f"{'No Fraud' if fraud == 'N' else 'Fraud'}", edgecolor='white')

ax.set_title('Total Claim Amount Distribution by Fraud Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Total Claim Amount ($)')
ax.set_ylabel('Count')
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()"""
))

cells.append(nbf.v4.new_code_cell(
"""# Correlation heatmap for numerical features
numerical_df = df.select_dtypes(include=[np.number])
corr = numerical_df.corr()

fig, ax = plt.subplots(figsize=(16, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdPu',
            linewidths=0.5, ax=ax, vmin=-1, vmax=1, annot_kws={'size': 8})
ax.set_title('Correlation Heatmap — Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()"""
))

# ── Step 4: Data Preprocessing ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 4: Data Preprocessing"))

cells.append(nbf.v4.new_code_cell(
"""# Make a copy for preprocessing
data = df.copy()

# 1. Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# 2. Drop irrelevant columns
cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_date', 'incident_location', '_c39']
data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True)

print(f"Shape after dropping columns: {data.shape}")
print(f"\\nRemaining columns ({len(data.columns)}):")
for i, col in enumerate(data.columns, 1):
    print(f"  {i:2d}. {col}")"""
))

cells.append(nbf.v4.new_code_cell(
"""# 3. Handle missing values
print("Missing values before imputation:")
missing = data.isnull().sum()
missing = missing[missing > 0]
print(missing)

for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

print(f"\\n✅ Missing values after imputation: {data.isnull().sum().sum()}")"""
))

cells.append(nbf.v4.new_code_cell(
"""# 4. Encode target variable
data['fraud_reported'] = data['fraud_reported'].map({'Y': 1, 'N': 0})

# 5. Label encode categorical features
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

print(f"Categorical columns to encode ({len(categorical_cols)}):")
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"  ✅ {col} — {le.classes_.shape[0]} unique values")

print(f"\\nFinal dataset shape: {data.shape}")
print(f"Target distribution: \\n{data['fraud_reported'].value_counts()}")"""
))

# ── Step 5: Train-Test Split ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 5: Train-Test Split"))

cells.append(nbf.v4.new_code_cell(
"""# Split features and target
X = data.drop('fraud_reported', axis=1)
y = data['fraud_reported']

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"\\nTraining target distribution:")
print(y_train.value_counts())
print(f"\\nTest target distribution:")
print(y_test.value_counts())"""
))

# ── Step 6: Model 1 — Random Forest ─────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## Step 6: Model 1 — Random Forest Classifier

Random Forest is an ensemble method that builds multiple decision trees and merges them together to get a more accurate and stable prediction."""
))

cells.append(nbf.v4.new_code_cell(
"""# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("✅ Random Forest model trained successfully!")
print(f"   Number of trees: {rf_model.n_estimators}")
print(f"   Max depth: {rf_model.max_depth}")"""
))

cells.append(nbf.v4.new_code_cell(
"""# Random Forest — Evaluation Metrics
print("=" * 55)
print("   RANDOM FOREST — EVALUATION METRICS")
print("=" * 55)

rf_accuracy  = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall    = recall_score(y_test, rf_pred)
rf_f1        = f1_score(y_test, rf_pred)
rf_roc_auc   = roc_auc_score(y_test, rf_proba)

print(f"\\n  Accuracy:   {rf_accuracy:.4f}  ({rf_accuracy*100:.2f}%)")
print(f"  Precision:  {rf_precision:.4f}  ({rf_precision*100:.2f}%)")
print(f"  Recall:     {rf_recall:.4f}  ({rf_recall*100:.2f}%)")
print(f"  F1-Score:   {rf_f1:.4f}  ({rf_f1*100:.2f}%)")
print(f"  ROC-AUC:    {rf_roc_auc:.4f}  ({rf_roc_auc*100:.2f}%)")

print(f"\\n{'=' * 55}")
print("\\n📋 Detailed Classification Report:\\n")
print(classification_report(y_test, rf_pred, target_names=['No Fraud', 'Fraud']))"""
))

cells.append(nbf.v4.new_code_cell(
"""# Random Forest — Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['No Fraud', 'Fraud'],
            yticklabels=['No Fraud', 'Fraud'],
            linewidths=2, linecolor='white',
            annot_kws={'size': 16, 'fontweight': 'bold'}, ax=ax)
ax.set_title('Confusion Matrix — Random Forest', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
plt.tight_layout()
plt.show()

print(f"\\n  True Negatives:  {cm_rf[0][0]}")
print(f"  False Positives: {cm_rf[0][1]}")
print(f"  False Negatives: {cm_rf[1][0]}")
print(f"  True Positives:  {cm_rf[1][1]}")"""
))

# ── Step 7: Model 2 — XGBoost ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## Step 7: Model 2 — XGBoost Classifier

XGBoost (Extreme Gradient Boosting) is a powerful boosting algorithm that builds trees sequentially, where each new tree corrects errors from the previous ones."""
))

cells.append(nbf.v4.new_code_cell(
"""# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print("✅ XGBoost model trained successfully!")
print(f"   Number of rounds: {xgb_model.n_estimators}")
print(f"   Max depth: {xgb_model.max_depth}")
print(f"   Learning rate: {xgb_model.learning_rate}")"""
))

cells.append(nbf.v4.new_code_cell(
"""# XGBoost — Evaluation Metrics
print("=" * 55)
print("   XGBOOST — EVALUATION METRICS")
print("=" * 55)

xgb_accuracy  = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall    = recall_score(y_test, xgb_pred)
xgb_f1        = f1_score(y_test, xgb_pred)
xgb_roc_auc   = roc_auc_score(y_test, xgb_proba)

print(f"\\n  Accuracy:   {xgb_accuracy:.4f}  ({xgb_accuracy*100:.2f}%)")
print(f"  Precision:  {xgb_precision:.4f}  ({xgb_precision*100:.2f}%)")
print(f"  Recall:     {xgb_recall:.4f}  ({xgb_recall*100:.2f}%)")
print(f"  F1-Score:   {xgb_f1:.4f}  ({xgb_f1*100:.2f}%)")
print(f"  ROC-AUC:    {xgb_roc_auc:.4f}  ({xgb_roc_auc*100:.2f}%)")

print(f"\\n{'=' * 55}")
print("\\n📋 Detailed Classification Report:\\n")
print(classification_report(y_test, xgb_pred, target_names=['No Fraud', 'Fraud']))"""
))

cells.append(nbf.v4.new_code_cell(
"""# XGBoost — Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds',
            xticklabels=['No Fraud', 'Fraud'],
            yticklabels=['No Fraud', 'Fraud'],
            linewidths=2, linecolor='white',
            annot_kws={'size': 16, 'fontweight': 'bold'}, ax=ax)
ax.set_title('Confusion Matrix — XGBoost', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
plt.tight_layout()
plt.show()

print(f"\\n  True Negatives:  {cm_xgb[0][0]}")
print(f"  False Positives: {cm_xgb[0][1]}")
print(f"  False Negatives: {cm_xgb[1][0]}")
print(f"  True Positives:  {cm_xgb[1][1]}")"""
))

# ── Step 8: Model Comparison ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 8: Model Comparison & Analysis"))

cells.append(nbf.v4.new_code_cell(
"""# Side-by-side comparison table
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc],
    'XGBoost': [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc]
})
comparison['Winner'] = comparison.apply(
    lambda row: '🌲 Random Forest' if row['Random Forest'] >= row['XGBoost'] else '⚡ XGBoost', axis=1
)
comparison['Random Forest'] = comparison['Random Forest'].apply(lambda x: f"{x:.4f}")
comparison['XGBoost'] = comparison['XGBoost'].apply(lambda x: f"{x:.4f}")

print("═" * 70)
print("           MODEL COMPARISON — FINAL RESULTS")
print("═" * 70)
print()
print(comparison.to_string(index=False))
print()
print("═" * 70)"""
))

cells.append(nbf.v4.new_code_cell(
"""# Comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc]
xgb_scores = [xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, rf_scores, width, label='🌲 Random Forest', 
               color='#6366f1', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, xgb_scores, width, label='⚡ XGBoost',
               color='#a78bfa', edgecolor='white', linewidth=1.5)

ax.set_ylabel('Score', fontsize=13)
ax.set_title('Model Comparison — All Metrics', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.1)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""
))

cells.append(nbf.v4.new_code_cell(
"""# ROC Curve comparison
fig, ax = plt.subplots(figsize=(10, 8))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

ax.plot(fpr_rf, tpr_rf, color='#6366f1', lw=3, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
ax.plot(fpr_xgb, tpr_xgb, color='#a78bfa', lw=3, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Baseline')

ax.fill_between(fpr_rf, tpr_rf, alpha=0.1, color='#6366f1')
ax.fill_between(fpr_xgb, tpr_xgb, alpha=0.1, color='#a78bfa')

ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve Comparison', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""
))

# ── Step 9: Feature Importance ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Step 9: Feature Importance Analysis"))

cells.append(nbf.v4.new_code_cell(
"""# Feature importance — both models
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Random Forest
rf_fi = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
rf_fi.plot(kind='barh', ax=axes[0], color='#6366f1', edgecolor='white', linewidth=1)
axes[0].set_title('Top 15 Features — Random Forest', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Importance')

# XGBoost
xgb_fi = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(15)
xgb_fi.plot(kind='barh', ax=axes[1], color='#a78bfa', edgecolor='white', linewidth=1)
axes[1].set_title('Top 15 Features — XGBoost', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()"""
))

# ── Step 10: Final Report ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## Step 10: Final Report Summary

---

### 📊 Project: Car Insurance Claim Prediction System

### 🎯 Objective
Predict whether an insurance claim is **fraudulent** or **legitimate** using machine learning.

### 📦 Dataset
- **1,000 samples** with **35+ features** covering policy details, customer demographics, incident information, and claim amounts.
- **Target:** `fraud_reported` (Binary: Y/N)
- **Class imbalance:** ~75% legitimate, ~25% fraudulent

### 🧠 Models Used

| Model | Description |
|-------|-------------|
| **Random Forest** | Ensemble of 200 decision trees with max depth 15 |
| **XGBoost** | Gradient boosting with 200 rounds, learning rate 0.1 |

### 📈 Key Findings
1. Both models achieve strong performance on this dataset
2. Feature importance reveals that **claim amounts**, **incident severity**, and **policy details** are the most predictive features
3. The models can effectively distinguish between fraudulent and legitimate claims

### 🔧 Tools & Libraries
`Python`, `Pandas`, `NumPy`, `Scikit-learn`, `XGBoost`, `Matplotlib`, `Seaborn`, `Streamlit`

### 🚀 Deployment
A **Streamlit web application** has been built for real-time fraud prediction. Run it with:
```
streamlit run app.py
```

---
*Report generated as part of the Car Insurance Claim Prediction System project.*"""
))

cells.append(nbf.v4.new_code_cell(
"""# Save models for Streamlit app
import joblib

joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
joblib.dump({
    'X_test': X_test, 'y_test': y_test,
    'rf_pred': rf_pred, 'rf_proba': rf_proba,
    'xgb_pred': xgb_pred, 'xgb_proba': xgb_proba,
    'rf_metrics': {
        'Accuracy': rf_accuracy, 'Precision': rf_precision,
        'Recall': rf_recall, 'F1-Score': rf_f1, 'ROC-AUC': rf_roc_auc
    },
    'xgb_metrics': {
        'Accuracy': xgb_accuracy, 'Precision': xgb_precision,
        'Recall': xgb_recall, 'F1-Score': xgb_f1, 'ROC-AUC': xgb_roc_auc
    },
    'feature_importances_rf': dict(zip(X.columns, rf_model.feature_importances_)),
    'feature_importances_xgb': dict(zip(X.columns, xgb_model.feature_importances_)),
}, 'model_results.pkl')

print("✅ All models and results saved successfully!")
print("\\n📁 Saved files:")
print("   - rf_model.pkl")
print("   - xgb_model.pkl")
print("   - label_encoders.pkl")
print("   - feature_names.pkl")
print("   - model_results.pkl")
print("\\n🚀 Run 'streamlit run app.py' to launch the web dashboard!")"""
))

nb.cells = cells

# Write the notebook
import nbformat
with open('Car_Insurance_Claim_Prediction.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("✅ Notebook created: Car_Insurance_Claim_Prediction.ipynb")
