import pickle
import os
from flask import Flask, request, jsonify

# -----------------------------
# 1. Load the saved model + vectorizer
# -----------------------------
model_file = os.environ.get("MODEL_FILE", "xgb_churn_pipeline.pkl")

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
# 3. Only run Flask server locally (not used in Render)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9696))
    app.run(debug=True, host='0.0.0.0', port=port)
