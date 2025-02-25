from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess training data
df_train = pd.read_csv('train_values.csv')
df_label = pd.read_csv('Train_Labels.csv')

df_train = pd.concat([df_train, df_label['heart_disease_present']], axis=1)
df_train['thal'] = df_train['thal'].map({'normal': 0, 'reversible_defect': 1, 'fixed_defect': 2})

# Select features for training
X = df_train[['chest_pain_type', 'max_heart_rate_achieved']].values
y = df_train.iloc[:, -1].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train logistic regression model
log_reg = LogisticRegression(class_weight='balanced', max_iter=300000)
log_reg.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df_patient = pd.DataFrame(data)

        df_patient['thal'] = df_patient['thal'].map({'normal': 0, 'reversible defect': 1, 'fixed defect': 2})

        X_patient = df_patient[['chest_pain_type', 'max_heart_rate_achieved']].values
        X_patient = scaler.transform(X_patient)

        predictions = log_reg.predict(X_patient).tolist()

        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
