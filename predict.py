import pickle
from flask import Flask, request, jsonify

# -----------------------------
# 1. Load the saved model + vectorizer
# -----------------------------
model_file = 'xgb_churn_pipeline.pkl'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

dv = pipeline['vectorizer']
model = pipeline['model']

# -----------------------------
# 2. Create Flask app
# -----------------------------
app = Flask('churn_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON input representing a customer, e.g.:
    {
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
    }
    """
    customer = request.get_json()

    # Preprocess input
    X = dv.transform([customer])

    # Predict churn probability
    churn_prob = model.predict_proba(X)[0, 1]
    churn = churn_prob >= 0.5  # threshold 0.5

    result = {
        'churn_probability': float(churn_prob),
        'churn': bool(churn)
    }

    return jsonify(result)

# -----------------------------
# 3. Run Flask app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
