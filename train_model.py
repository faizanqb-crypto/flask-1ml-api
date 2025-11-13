from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1️⃣ Train and save the model
data = pd.DataFrame({
    'age': [25, 45, 31, 35, 22, 60, 48, 50, 29, 41],
    'income': [20000, 50000, 32000, 40000, 18000, 70000, 52000, 60000, 25000, 45000],
    'loan_amount': [1000, 15000, 5000, 7000, 800, 20000, 12000, 16000, 2000, 9000],
    'default': [0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
})
X=data[['age', 'income', 'loan_amount']]
y=data[['default']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("test accuracy: ", accuracy_score(y_test, preds))

with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)
print("Model saved successfully")

