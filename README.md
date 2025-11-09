# Netflix Customer Churn Prediction

---

## **1. Problem Description**

I wants to predict whether a  Netflix customer is likely to churn (cancel their subscription) based on demographics, subscription details, and usage patterns.  

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
