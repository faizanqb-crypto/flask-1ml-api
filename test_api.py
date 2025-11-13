import requests
import json

# Load JSON
with open("test.json") as f:
    data = json.load(f)

url = "http://127.0.0.1:5000/predict"
response = requests.post(url, json=data)
try:
    print("Parsed response:", response.json())
except Exception as e:
    print("JSON decode error:", e)