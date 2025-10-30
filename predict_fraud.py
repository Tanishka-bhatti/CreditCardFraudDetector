import joblib
import pandas as pd
from datetime import datetime

# === Load trained model and preprocessing tools ===
data = joblib.load('kaggle_fraud_model.pkl')
model = data['model']
scaler = data['scaler']
label_encoders = data['label_encoders']
feature_columns = data['feature_columns']

# === Create a new (example) transaction ===
new_transaction = pd.DataFrame([{
    'Unnamed: 0': 0,
    'trans_date_trans_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'cc_num': 1234567890123456,
    'merchant': 'fraud_JohnDoeShop',
    'category': 'shopping_pos',
    'amt': 154.75,
    'first': 'Alice',
    'last': 'Smith',
    'gender': 'F',
    'street': '123 Elm Street',
    'city': 'Charlotte',
    'state': 'NC',
    'zip': 28202,
    'lat': 35.2271,
    'long': -80.8431,
    'city_pop': 872498,
    'job': 'Engineer',
    'dob': '1988-06-15',
    'trans_num': '123abc456def789',
    'unix_time': 1371816893,
    'merch_lat': 35.2300,
    'merch_long': -80.8500
}])

# === Apply label encoding to categorical features ===
for col, le in label_encoders.items():
    if col in new_transaction.columns:
        try:
            new_transaction[col] = le.transform(new_transaction[col])
        except ValueError:
            # Handle unseen labels
            new_transaction[col] = [0]

# === Scale numeric data ===
new_transaction_scaled = scaler.transform(new_transaction[feature_columns])

# === Predict ===
prediction = model.predict(new_transaction_scaled)[0]
probability = model.predict_proba(new_transaction_scaled)[0][1]

print(f"\n🚨 Prediction: {'FRAUD' if prediction == 1 else 'NOT FRAUD'}")
print(f"🧠 Fraud Probability: {probability:.4f}\n")
