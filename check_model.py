import joblib

models = joblib.load('kaggle_fraud_model.pkl')
print("Type of loaded object:", type(models))

if isinstance(models, dict):
    print("Available keys:", models.keys())
else:
    print("Loaded model:", models)
data = joblib.load('kaggle_fraud_model.pkl')
print("Feature columns used during training:\n")
print(data['feature_columns'])