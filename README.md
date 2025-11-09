# Netflix Customer Churn Prediction

---

## **1. Problem Description**

Churn prediction is critical for subscription-based services like Netflix. The goal of this project is to predict which customers are likely to churn using demographic, behavioral, and subscription data. Accurate predictions allow proactive retention strategies, improving revenue and customer satisfaction.

Dataset: Netflix Customer Churn Prediction 

**Goal:** Build a machine learning model to estimate churn probability for each customer to enable proactive retention strategies.

---
## **2. Data Preparation & Cleaning**

- **Removed outliers in avg_watch_time_per_day (max valid value = 24 hours).**
- **Dropped monthly_fee because it duplicates subscription_type.**
- **Converted numeric age to categorical age_group to capture non-linear effects on churn:
18–25, 26–35, 36–45, 46–60, 60+.**
  The original numeric age has a very weak correlation with churn (-0.003)
<img width="571" height="455" alt="output_47_0" src="https://github.com/user-attachments/assets/b91e0e4e-31b0-470f-b1b4-e7f879e8a148" />
- **Removed customer_id (identifier, not predictive).**
- **No missing values in dataset.**
Categorical Features: gender, region, subscription_type, device, payment_method, favorite_genre, age_group
Numerical Features: watch_hours, last_login_days, number_of_profiles, avg_watch_time_per_day

## **3. Exploratory Data Analysis (EDA)**
- **Target variable: Churned vs Not Churned (≈50/50 balanced).**
- **Categorical Analysis:**
  Subscription Type: Basic users churn most (~62%), Premium least (~44%).
  Payment Method: Crypto & Gift Card users churn more (~60%), Debit/Credit less (~44%).
- **Numerical Analysis:**
  Strong churn drivers: low engagement (watch_hours, avg_watch_time_per_day) and recency (last_login_days).
  Weak influence: age, number_of_profiles.
  <img width="798" height="689" alt="output_42_0" src="https://github.com/user-attachments/assets/4b306fd6-89a8-4f10-9e0c-93650d62f026" />

- **Mutual Information Ranking: subscription_type > payment_method > others**

---

## **4. Model Training**
- **Baseline:** Logistic Regression (F1-score ~0.91, ROC-AUC ~0.97).  
- **Advanced Models:** Random Forest and XGBoost.  
- **Parameter Tuning:** Grid search and cross-validation used to optimize hyperparameters.  

### Model Comparison

| Model                   | F1-score | ROC-AUC | Notes                                           |
|-------------------------|----------|---------|------------------------------------------------|
| Logistic Regression     | 0.905    | 0.971   | Simple baseline, interpretable coefficients  |
| Random Forest           | 0.994    | 0.998   | High accuracy, robust generalization         |
| XGBoost                 | 0.994    | 0.999   | Best overall performance and stability      |

**Selected Model:** XGBoost, based on highest accuracy, F1-score, and ROC-AUC.

---

## **5. Exporting Model**
- Trained XGBoost pipeline is exported using `pickle`.  
- Includes both the vectorizer (for preprocessing) and the trained model.  
- Ensures the model can be reloaded and used for predictions in a reproducible manner.

---

## **6. Reproducibility**
- Notebook and training script can be executed without errors.  
- Dataset is publicly available on Kaggle: [Netflix Customer Churn Prediction](https://www.kaggle.com/datasets/vasifasad/netflix-customer-churn-prediction).  
- Environment dependencies captured in `environment.yml` (conda) for easy setup.

---

## **7. Model Deployment**

### Local Deployment
- **Framework:** Flask  
- **Usage:**  
  - API endpoint `/predict` accepts JSON input for a customer.  
  - Returns churn probability and binary churn prediction.  
- **Run Command:**  
```bash
python predict.py
