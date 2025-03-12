from flask import Flask, request, jsonify
import pandas as pd
import requests
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from model import process_patient_data 

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb://mongodb:27017/")  # Change if needed (e.g., localhost)
db = client["openemr"]
predictions_collection = db["predictions"]

def wait_for_openemr():
    url = "http://openemr:8300"  # Ensure this is the correct URL and port
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("OpenEMR is up and running")
                break
        except requests.ConnectionError:
            print("Waiting for OpenEMR...")
        time.sleep(5)

wait_for_openemr()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Process the data and make predictions
        process_patient_data(data)  # Process and insert predictions
        return jsonify({'message': 'Prediction process completed.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

