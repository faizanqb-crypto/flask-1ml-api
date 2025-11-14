from flask import Flask, request, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return "LOAN PREDICTION API is up"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print raw data for debugging
        print("Raw request data:", request.data)

        # Parse JSON safely
        data = request.get_json(force=True)
        print("Parsed JSON:", data)

        # Ensure it's a list
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        expected_cols = ['age', 'income', 'loan_amount']
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400

        preds = model.predict(df[expected_cols])
        probs = model.predict_proba(df[expected_cols]).tolist()

        return jsonify({'prediction': preds.tolist(), 'probabilities': probs})

    except Exception as e:
        # Print error in Flask console
        print("Error:", e)
        # Always return JSON even on error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)