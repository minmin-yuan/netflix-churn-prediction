import kagglehub
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import pickle

# -----------------------------
# 1. Load dataset
# -----------------------------
path = kagglehub.dataset_download("vasifasad/netflix-customer-churn-prediction")
df = pd.read_csv(os.path.join(path, "netflix_customer_churn.csv"))

# Drop redundant column
df = df.drop('monthly_fee', axis=1)

# -----------------------------
# 2. Feature engineering
# -----------------------------
# Bin age into categories
bins = [18, 25, 35, 45, 60, 100]
labels = ['18–25', '26–35', '36–45', '46–60', '60+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Define categorical and numerical features
categorical = ['gender', 'region', 'subscription_type', 'device', 'payment_method', 'favorite_genre', 'age_group']
numerical = ['watch_hours', 'last_login_days', 'number_of_profiles', 'avg_watch_time_per_day']

# -----------------------------
# 3. Split data
# -----------------------------
df_train, df_test = train_test_split(df, test_size=0.2, random_state=10)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churned.values
y_test = df_test.churned.values

del df_train['churned']
del df_test['churned']

# -----------------------------
# 4. Preprocessing
# -----------------------------
dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

test_dict = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dict)

# -----------------------------
# 5. Train XGBoost model
# -----------------------------
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=10,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate model
# -----------------------------
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7. Save model + vectorizer
# -----------------------------
with open("xgb_churn_pipeline.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": dv,
        "model": xgb_model
    }, f)

print("Model and vectorizer saved successfully!")
