import requests
import json

url = "http://127.0.0.1:9696/predict"

# Example customer data
customer = {
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

# Send POST request
response = requests.post(url, json=customer)

# Print response
if response.status_code == 200:
    print("Response from API:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
