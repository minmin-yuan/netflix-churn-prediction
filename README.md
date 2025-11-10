# Netflix Customer Churn Prediction

---

## **1. Problem Description**

Churn prediction is critical for subscription-based services like Netflix. The goal of this project is to predict which customers are likely to churn using demographic, behavioral, and subscription data. Accurate predictions allow proactive retention strategies, improving revenue and customer satisfaction.

- Dataset is publicly available on Kaggle: [Netflix Customer Churn Prediction](https://www.kaggle.com/datasets/vasifasad/netflix-customer-churn-prediction).  


---
## **2. Data Preparation & Cleaning**

- **Removed outliers in avg_watch_time_per_day (max valid value = 24 hours).**
- **Dropped monthly_fee because it duplicates subscription_type.**
- **Converted numeric age to categorical age_group to capture non-linear effects on churn:**\
  18–25, 26–35, 36–45, 46–60, 60+.
  The original numeric age has a very weak correlation with churn (-0.003)
<img width="571" height="455" alt="output_47_0" src="https://github.com/user-attachments/assets/b91e0e4e-31b0-470f-b1b4-e7f879e8a148" />

- **Removed customer_id (identifier, not predictive).**
- **No missing values in dataset.**
- **Categorical Features:**\
  gender, region, subscription_type, device, payment_method, favorite_genre, age_group
- **Numerical Features:** \
  watch_hours, last_login_days, number_of_profiles, avg_watch_time_per_day

## **3. Exploratory Data Analysis (EDA)**
- **Target variable: Churned vs Not Churned (≈50/50 balanced).**
- **Categorical Analysis:**\
  Subscription Type: Basic users churn most (~62%), Premium least (~44%).\
  <img width="590" height="290" alt="output_37_2" src="https://github.com/user-attachments/assets/9069cf30-dd70-4ac2-8692-6c0faeeb6c18" />

  Payment Method: Crypto & Gift Card users churn more (~60%), Debit/Credit less (~44%).
  <img width="590" height="290" alt="output_37_4" src="https://github.com/user-attachments/assets/5318b412-facc-41e8-bd06-fad96b8b34ca" />

- **Numerical Analysis:**\
  - Strong churn drivers: 
    low engagement (watch_hours, avg_watch_time_per_day) and recency (last_login_days).
  - Weak influence: 
    age(which is why we replaced it with categorical age group as mentioned before, but here the correlation heatmap was created before the binning step)
  <img width="798" height="689" alt="output_42_0" src="https://github.com/user-attachments/assets/4b306fd6-89a8-4f10-9e0c-93650d62f026" />

- **Mutual Information Ranking: subscription_type > payment_method > others**

| Feature           | Mutual Information |
|------------------|------------------|
| subscription_type | 0.013533         |
| payment_method    | 0.009823         |
| favorite_genre    | 0.000639         |
| region            | 0.000286         |
| device            | 0.000143         |
| gender            | 0.000074         |


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
- Environment dependencies captured in `environment.yml` (conda) for easy setup or `requirements.txt` (pip + venv).

---

## **7. Model Deployment**

Run the service locally using Flask or Gunicorn. This section covers environment setup, installing dependencies, preparing the trained model, running the server, and testing.

### Local Deployment 
#### 1. Prepare environment

**Option A — Conda**

```bash
# create environment from provided file
conda env create -f environment.yml

# activate it
conda activate netflix-churn
```

**Option B — pip + venv**
create and activate a venv
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

Windows (PowerShell)
```bash
.venv\Scripts\activate
```

install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Prepare the trained model
To train and save the model locally:
```bash
python train.py
```
#### 3. Run the service
- **Run the Flask service locally**
```bash
python predict.py
```
Defalut port:9696\
Access locally: http://127.0.0.1:9696/predict

- **Run the service (production-like) with Gunicorn**
```bash
gunicorn --bind 0.0.0.0:9696 predict:app --workers 4
```
#### 4.Test the API: Use curl or a tool like Postman:**
```
curl -X POST http://127.0.0.1:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34,
    "gender": "Female",
    "subscription_type": "Standard",
    "watch_hours": 15.3,
    "last_login_days": 5,
    "region": "Europe",
    "device": "Mobile",
    "payment_method": "PayPal",
    "number_of_profiles": 3,
    "avg_watch_time_per_day": 1.2,
    "favorite_genre": "Drama",
    "age_group": "26–35"
  }'
```
- **Output: Churn probability and binary churn prediction.**


  
### Containerized Deployment (Docker)
The application can be containerized using Docker\
Dockerfile: Provided in the repository.

```
docker build -t netflix-churn .
```
```
docker run -p 9696:9696 netflix-churn
```
- screenshots of docker running and test
<img width="1356" height="292" alt="docker" src="https://github.com/user-attachments/assets/b413dbb4-06c4-4fed-84da-c9d0c19b9fab" />
<img width="1128" height="218" alt="test_api" src="https://github.com/user-attachments/assets/13617f43-37d5-4b21-97c2-bc9684780e04" />

### Cloud Deployment (Render)
- The service can be deployed to the cloud for remote access:
- Instructions for Render deployment (which is free for limited usage):
  1. push project to github repo
  2. Sign up(use github account is the easist) and log in to Render
  3. Once logged in, click New → Web Service
  4. Connect to github repo and select netflix-churn-prediction repo
  5. Click Create Web Service, Render will automatically build docker image, install dependencies from requirements.txt and start
  the Gunicorn server.
  6. After a minute or two, you’ll get a public URL like: https://netflix-churn-prediction.onrender.com
- Service URL: https://netflix-churn-prediction.onrender.com/predict (already deployed and can be accessd)
- Access: Send HTTP POST requests with JSON payloads as shown above.
- screenshots of Render deployment
<img width="2122" height="1152" alt="render cloud deployment" src="https://github.com/user-attachments/assets/2dc912f4-61d1-408a-bc25-76b9819ce4ee" />
<img width="1332" height="532" alt="render test" src="https://github.com/user-attachments/assets/cd9849ef-37da-4061-9adb-dec62f0a0373" />
<img width="1220" height="878" alt="Snip20251110_7" src="https://github.com/user-attachments/assets/1bf8fc43-159a-4d0a-8d98-171b95e3ca54" />
