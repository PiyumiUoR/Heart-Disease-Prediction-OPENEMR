import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess training data
df_train = pd.read_csv('train_values.csv')
df_label = pd.read_csv('Train_Labels.csv')

df_train = pd.concat([df_train, df_label['heart_disease_present']], axis=1)

# Encode categorical variables
df_train['thal'] = df_train['thal'].map({'normal': 0, 'reversible_defect': 1, 'fixed_defect': 2})

# Select features for training
X = df_train[['chest_pain_type', 'max_heart_rate_achieved']].values
y = df_train.iloc[:, -1].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
log_reg = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, solver='lbfgs', 
                             tol=0.00001, C=1.0, fit_intercept=True, max_iter=30000000)
log_reg.fit(X_train, y_train)

def fetch_patient_data(api_url, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        patient_data = response.json()

        if isinstance(patient_data, dict) and "data" in patient_data:
            patient_data = patient_data["data"]
        
        if not isinstance(patient_data, list):
            print("Error: Expected a list but got:", type(patient_data))
            return None, None

        df_patient = pd.DataFrame(patient_data)
        # print("df_patient: ", df_patient)

        df_patient = df_patient.where(pd.notna(df_patient), None)

        required_columns = ['uuid', 'chest_pain_type', 'max_heart_rate_achieved', 'thal']
        missing_columns = [col for col in required_columns if col not in df_patient.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None, None

        # Convert 'thal' if present
        df_patient['thal'] = df_patient['thal'].map({'normal': 0, 'reversible defect': 1, 'fixed defect': 2})

        # Extract required features for prediction
        X_patient = df_patient[['chest_pain_type', 'max_heart_rate_achieved']].values
        X_patient = scaler.transform(X_patient)  # Ensure 'scaler' is already fitted

        return df_patient, X_patient
    else:
        print(f"Error fetching patient data. Status code: {response.status_code}")
        return None, None

# Function to insert prediction and update heart_disease_present
# Function to insert prediction and update heart_disease_present
def insert_prediction(api_url, token, df_patient, patient_uuids, predictions):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    if isinstance(df_patient, pd.DataFrame):
        df_patient['heart_disease_present'] = predictions
        df_patient['uuid'] = df_patient['uuid'].astype(str)

        # **Replace NaN values with None (which translates to null in JSON)**
        df_patient = df_patient.replace({np.nan: None})

        json_payload = df_patient.to_dict(orient="records")
    else:
        print("Error: patient_data is not a DataFrame.")
        return

    for record in json_payload:
        uuid = record['uuid']
        patient_url = f"{api_url}/{uuid}"

        try:
            response = requests.put(patient_url, headers=headers, json=record)
            if response.status_code in [200, 201]:
                print(f"Successfully inserted prediction for patient {uuid}!")
            else:
                print(f"Error inserting prediction for patient {uuid}. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Request failed for patient {uuid}: {e}")


# Fetch and predict patient status
api_url = "http://86.50.231.152:8300/apis/default/api/patient"
bearer_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJQbFZoRDdNT3dYRW1iNXZ1VHFLYUVWRmR0dDRRejdVeDlxZW5JYmF2WGVBIiwianRpIjoiNGYwM2EzNzY5MzhjNDMyMTM2YmUxZGI4NThiZDA2MmM0OWI0YTI5YjkxODBhYWU3NDg4YmExNDAyZWNiYjZjMGI1ZDQxOTZkNGM4NTk0ZDgiLCJpYXQiOjE3NDA0NDM0MTkuMDQxMjU2LCJuYmYiOjE3NDA0NDM0MTkuMDQxMjYsImV4cCI6MTc0MDQ0NzAxOC45NjMwNTgsInN1YiI6IjllMTViNzAxLWNjMjgtNDMxNi05MmFiLTY4NDc5N2RlNGI1MCIsInNjb3BlcyI6WyJvcGVuaWQiLCJvZmZsaW5lX2FjY2VzcyIsImFwaTpvZW1yIiwiYXBpOmZoaXIiLCJhcGk6cG9ydCIsInVzZXIvYWxsZXJneS5yZWFkIiwidXNlci9hbGxlcmd5LndyaXRlIiwidXNlci9hcHBvaW50bWVudC5yZWFkIiwidXNlci9hcHBvaW50bWVudC53cml0ZSIsInVzZXIvZGVudGFsX2lzc3VlLnJlYWQiLCJ1c2VyL2RlbnRhbF9pc3N1ZS53cml0ZSIsInVzZXIvZG9jdW1lbnQucmVhZCIsInVzZXIvZG9jdW1lbnQud3JpdGUiLCJ1c2VyL2RydWcucmVhZCIsInVzZXIvZW5jb3VudGVyLnJlYWQiLCJ1c2VyL2VuY291bnRlci53cml0ZSIsInVzZXIvZmFjaWxpdHkucmVhZCIsInVzZXIvZmFjaWxpdHkud3JpdGUiLCJ1c2VyL2ltbXVuaXphdGlvbi5yZWFkIiwidXNlci9pbnN1cmFuY2UucmVhZCIsInVzZXIvaW5zdXJhbmNlLndyaXRlIiwidXNlci9pbnN1cmFuY2VfY29tcGFueS5yZWFkIiwidXNlci9pbnN1cmFuY2VfY29tcGFueS53cml0ZSIsInVzZXIvaW5zdXJhbmNlX3R5cGUucmVhZCIsInVzZXIvbGlzdC5yZWFkIiwidXNlci9tZWRpY2FsX3Byb2JsZW0ucmVhZCIsInVzZXIvbWVkaWNhbF9wcm9ibGVtLndyaXRlIiwidXNlci9tZWRpY2F0aW9uLnJlYWQiLCJ1c2VyL21lZGljYXRpb24ud3JpdGUiLCJ1c2VyL21lc3NhZ2Uud3JpdGUiLCJ1c2VyL3BhdGllbnQucmVhZCIsInVzZXIvcGF0aWVudC53cml0ZSIsInVzZXIvcHJhY3RpdGlvbmVyLnJlYWQiLCJ1c2VyL3ByYWN0aXRpb25lci53cml0ZSIsInVzZXIvcHJlc2NyaXB0aW9uLnJlYWQiLCJ1c2VyL3Byb2NlZHVyZS5yZWFkIiwidXNlci9zb2FwX25vdGUucmVhZCIsInVzZXIvc29hcF9ub3RlLndyaXRlIiwidXNlci9zdXJnZXJ5LnJlYWQiLCJ1c2VyL3N1cmdlcnkud3JpdGUiLCJ1c2VyL3RyYW5zYWN0aW9uLnJlYWQiLCJ1c2VyL3RyYW5zYWN0aW9uLndyaXRlIiwidXNlci92aXRhbC5yZWFkIiwidXNlci92aXRhbC53cml0ZSIsInVzZXIvQWxsZXJneUludG9sZXJhbmNlLnJlYWQiLCJ1c2VyL0NhcmVUZWFtLnJlYWQiLCJ1c2VyL0NvbmRpdGlvbi5yZWFkIiwidXNlci9Db3ZlcmFnZS5yZWFkIiwidXNlci9FbmNvdW50ZXIucmVhZCIsInVzZXIvSW1tdW5pemF0aW9uLnJlYWQiLCJ1c2VyL0xvY2F0aW9uLnJlYWQiLCJ1c2VyL01lZGljYXRpb24ucmVhZCIsInVzZXIvTWVkaWNhdGlvblJlcXVlc3QucmVhZCIsInVzZXIvT2JzZXJ2YXRpb24ucmVhZCIsInVzZXIvT3JnYW5pemF0aW9uLnJlYWQiLCJ1c2VyL09yZ2FuaXphdGlvbi53cml0ZSIsInVzZXIvUGF0aWVudC5yZWFkIiwidXNlci9QYXRpZW50LndyaXRlIiwidXNlci9QcmFjdGl0aW9uZXIucmVhZCIsInVzZXIvUHJhY3RpdGlvbmVyLndyaXRlIiwidXNlci9QcmFjdGl0aW9uZXJSb2xlLnJlYWQiLCJ1c2VyL1Byb2NlZHVyZS5yZWFkIiwicGF0aWVudC9lbmNvdW50ZXIucmVhZCIsInBhdGllbnQvcGF0aWVudC5yZWFkIiwicGF0aWVudC9BbGxlcmd5SW50b2xlcmFuY2UucmVhZCIsInBhdGllbnQvQ2FyZVRlYW0ucmVhZCIsInBhdGllbnQvQ29uZGl0aW9uLnJlYWQiLCJwYXRpZW50L0NvdmVyYWdlLnJlYWQiLCJwYXRpZW50L0VuY291bnRlci5yZWFkIiwicGF0aWVudC9JbW11bml6YXRpb24ucmVhZCIsInBhdGllbnQvTWVkaWNhdGlvblJlcXVlc3QucmVhZCIsInBhdGllbnQvT2JzZXJ2YXRpb24ucmVhZCIsInBhdGllbnQvUGF0aWVudC5yZWFkIiwicGF0aWVudC9Qcm9jZWR1cmUucmVhZCIsInNpdGU6ZGVmYXVsdCJdfQ.vxqKijDvAWItrnuiI1oC8QYIPNWcJPnBi8smhaV6hQ9k-GB2drXDx2X5We0xiBMDmsYLZLAHjcw_-PjefCvRO3CVzmWtt4bWSFP2TzTA-M1PrR9vnmOraxBF7IyUoDa56gxJJJ_wOELhetFpmpvCs6TcXBaXIPMY9lyqQTTt-X_MDnPEx-UUuNc_5QsBHsVQ3uhPj2dR3pUl1EVp2EDvXRSWqBuTJlsHmrM7V-aKY3FqbCoL6ARSP7xMJZkTTa6QMyxFZkH8wwGdlFn5MErDGb1AIoBU9UZVbpwKaYwbEBSGyE7o2jiBcKpWMTnxcaTf-GzoWBf5EE9MnzOYxfhByA"

df_patient, patient_uuids = fetch_patient_data(api_url, bearer_token)

if df_patient is not None and patient_uuids is not None:
    predictions = log_reg.predict(df_patient[['chest_pain_type', 'max_heart_rate_achieved']].values)  # Generate predictions
    
    # Convert predictions to a list of integers (for JSON serialization)
    predictions = predictions.tolist()

    # Insert predictions back into the database
    insert_prediction(api_url, bearer_token, df_patient, patient_uuids, predictions)
else:
    print("Failed to fetch patient data for prediction.")
