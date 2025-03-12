import pandas as pd
import json
import requests
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from kafka import KafkaConsumer

# Connect to MongoDB
client = MongoClient("mongodb://mongodb:27017/")  # Use "localhost" if running outside Docker
db = client["openemr"]
predictions_collection = db["predictions"]

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

def fetch_patient_data_from_message(message):
    # Convert the Kafka message value (which is in bytes) to a dictionary
    patient_data = json.loads(message)  # assuming the message is a JSON string
    
    # Convert to DataFrame (assuming the message contains the patient data in the appropriate structure)
    df_patient = pd.DataFrame([patient_data])
    
    # Handle missing data or NaNs
    df_patient = df_patient.where(pd.notna(df_patient), None)
    
    required_columns = ['uuid', 'chest_pain_type', 'max_heart_rate_achieved', 'thal']
    missing_columns = [col for col in required_columns if col not in df_patient.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None, None

    # Convert 'thal' if present in the data
    df_patient['thal'] = df_patient['thal'].map({'normal': 0, 'reversible defect': 1, 'fixed defect': 2})
    
    # Extract features for prediction (scaling as well)
    X_patient = df_patient[['chest_pain_type', 'max_heart_rate_achieved']].values
    
    # Ensure the scaler used in training is applied to the new data
    X_patient = scaler.transform(X_patient)
    
    return df_patient, X_patient

def process_patient_data(data):
    # Preprocessing and prediction logic
    df_patient, X_patient = fetch_patient_data_from_message(data)  
    if df_patient is not None and X_patient is not None:
        predictions = log_reg.predict(X_patient).tolist()  # Generate predictions
        
        # Insert predictions back into MongoDB
        insert_prediction(df_patient, predictions)

def insert_prediction(df_patient, predictions):
    reverse_map = {0: 'normal', 1: 'reversible defect', 2: 'fixed defect'}
    df_patient['thal'] = df_patient['thal'].map(reverse_map)

    if isinstance(df_patient, pd.DataFrame):
        df_patient['heart_disease_present'] = predictions
        df_patient['uuid'] = df_patient['uuid'].astype(str)
        df_patient = df_patient.replace({np.nan: None})
        json_payload = df_patient.to_dict(orient="records")
    else:
        print("Error: patient_data is not a DataFrame.")
        return

    for record in json_payload:
        uuid = record['uuid']
        patient_url = f"http://86.50.231.152:8300/apis/default/api/patient/{uuid}"

        try:
            response = requests.put(patient_url, headers={"Content-Type": "application/json"}, json=record)
            if response.status_code in [200, 201]:
                print(f"Successfully inserted prediction for patient {uuid}!")
                # Store prediction in MongoDB
                record['timestamp'] = pd.Timestamp.now()
                predictions_collection.insert_one(record)
            else:
                print(f"Error inserting prediction for patient {uuid}. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Request failed for patient {uuid}: {e}")

# Set up Kafka Consumer
consumer = KafkaConsumer(
    'patient_data_updates',  # Topic where patient data is streamed
    bootstrap_servers=['kafka-1:9092', 'kafka-2:9093', 'kafka-3:9094'],
    group_id='ml-service-group',
    auto_offset_reset='earliest'  # Start consuming from the beginning
)

for message in consumer:
    patient_data = message.value 
    process_patient_data(patient_data)
