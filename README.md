# Netflix Customer Churn Prediction

![Python Version](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## **1. Problem Description**

Netflix wants to predict whether a customer is likely to churn (cancel their subscription) based on demographics, subscription details, and usage patterns.  

**Goal:** Build a machine learning model to estimate churn probability for each customer to enable proactive retention strategies.

---

## **2. Exploratory Data Analysis (EDA)**

- **Dataset Features:** Age, gender, subscription type, device, region, watch hours, last login days, payment method, number of profiles, favorite genre, and average watch time per day.  
- **Missing Values:** Checked and no missing values found.  
- **Target Variable:** `churned` (0 = active, 1 = churned).  
- **Feature Analysis:**

| Feature | Notes |
|---------|-------|
| Numerical | `watch_hours`, `last_login_days`, `avg_watch_time_per_day`, `number_of_profiles` |
| Categorical | `gender`, `region`, `subscription_type`, `device`, `payment_method`, `favorite_genre` |

- **Visuals:**  

![Watch Hours Distribution](images/watch_hours_distribution.png)  
*Distribution of watch hours across all users.*

![Churn by Subscription Type](images/churn_by_subsc)
