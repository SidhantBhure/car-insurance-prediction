"""
Train Models Script - Car Insurance Claim Prediction System
Trains Random Forest and XGBoost models and saves them for Streamlit app.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='insurance_claims.csv'):
    """Load and preprocess the insurance claims dataset."""
    df = pd.read_csv(filepath)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)
    
    # Drop irrelevant columns
    cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_date', 
                    'incident_location', '_c39']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Encode target variable
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
    
    # Label encode categorical features
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    return df, label_encoders, categorical_cols

def train_and_evaluate():
    """Train both models and return results."""
    df, label_encoders, categorical_cols = load_and_preprocess_data()
    
    X = df.drop('fraud_reported', axis=1)
    y = df['fraud_reported']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ---- Model 1: Random Forest ----
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_metrics = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'F1-Score': f1_score(y_test, rf_pred),
        'ROC-AUC': roc_auc_score(y_test, rf_proba)
    }
    
    # ---- Model 2: XGBoost ----
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    xgb_metrics = {
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1-Score': f1_score(y_test, xgb_pred),
        'ROC-AUC': roc_auc_score(y_test, xgb_proba)
    }
    
    # Save models and artifacts
    joblib.dump(rf_model, 'rf_model.pkl')
    joblib.dump(xgb_model, 'xgb_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
    joblib.dump({
        'X_test': X_test, 'y_test': y_test,
        'rf_pred': rf_pred, 'rf_proba': rf_proba,
        'xgb_pred': xgb_pred, 'xgb_proba': xgb_proba,
        'rf_metrics': rf_metrics, 'xgb_metrics': xgb_metrics,
        'feature_importances_rf': dict(zip(X.columns, rf_model.feature_importances_)),
        'feature_importances_xgb': dict(zip(X.columns, xgb_model.feature_importances_)),
    }, 'model_results.pkl')
    
    print("=" * 60)
    print("  CAR INSURANCE CLAIM PREDICTION - TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n{'Metric':<15} {'Random Forest':>15} {'XGBoost':>15}")
    print("-" * 45)
    for metric in rf_metrics:
        print(f"{metric:<15} {rf_metrics[metric]:>15.4f} {xgb_metrics[metric]:>15.4f}")
    print("-" * 45)
    print("\nModels saved: rf_model.pkl, xgb_model.pkl")
    print("Results saved: model_results.pkl")
    
    return rf_model, xgb_model, rf_metrics, xgb_metrics

if __name__ == '__main__':
    train_and_evaluate()
