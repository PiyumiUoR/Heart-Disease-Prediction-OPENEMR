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
def insert_prediction(api_url, token, df_patient, patient_uuids, predictions):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

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
bearer_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJQbFZoRDdNT3dYRW1iNXZ1VHFLYUVWRmR0dDRRejdVeDlxZW5JYmF2WGVBIiwianRpIjoiZmNmMjViYzc5OWFiYzBjYzkxODU0MmUyNzM4NDhiNzI3N2NjMjdhYTc5NjA3YWJhMzBiM2UwOGY4YTQ4ZmEzMTYyZDY3NDE3YTJlNmJkNTIiLCJpYXQiOjE3NDA5NzA2NDguMzkzMzgsIm5iZiI6MTc0MDk3MDY0OC4zOTMzODgsImV4cCI6MTc0MDk3NDI0OC4zMTQ4Niwic3ViIjoiOWUxNWI3MDEtY2MyOC00MzE2LTkyYWItNjg0Nzk3ZGU0YjUwIiwic2NvcGVzIjpbIm9wZW5pZCIsIm9mZmxpbmVfYWNjZXNzIiwiYXBpOm9lbXIiLCJhcGk6ZmhpciIsImFwaTpwb3J0IiwidXNlci9hbGxlcmd5LnJlYWQiLCJ1c2VyL2FsbGVyZ3kud3JpdGUiLCJ1c2VyL2FwcG9pbnRtZW50LnJlYWQiLCJ1c2VyL2FwcG9pbnRtZW50LndyaXRlIiwidXNlci9kZW50YWxfaXNzdWUucmVhZCIsInVzZXIvZGVudGFsX2lzc3VlLndyaXRlIiwidXNlci9kb2N1bWVudC5yZWFkIiwidXNlci9kb2N1bWVudC53cml0ZSIsInVzZXIvZHJ1Zy5yZWFkIiwidXNlci9lbmNvdW50ZXIucmVhZCIsInVzZXIvZW5jb3VudGVyLndyaXRlIiwidXNlci9mYWNpbGl0eS5yZWFkIiwidXNlci9mYWNpbGl0eS53cml0ZSIsInVzZXIvaW1tdW5pemF0aW9uLnJlYWQiLCJ1c2VyL2luc3VyYW5jZS5yZWFkIiwidXNlci9pbnN1cmFuY2Uud3JpdGUiLCJ1c2VyL2luc3VyYW5jZV9jb21wYW55LnJlYWQiLCJ1c2VyL2luc3VyYW5jZV9jb21wYW55LndyaXRlIiwidXNlci9pbnN1cmFuY2VfdHlwZS5yZWFkIiwidXNlci9saXN0LnJlYWQiLCJ1c2VyL21lZGljYWxfcHJvYmxlbS5yZWFkIiwidXNlci9tZWRpY2FsX3Byb2JsZW0ud3JpdGUiLCJ1c2VyL21lZGljYXRpb24ucmVhZCIsInVzZXIvbWVkaWNhdGlvbi53cml0ZSIsInVzZXIvbWVzc2FnZS53cml0ZSIsInVzZXIvcGF0aWVudC5yZWFkIiwidXNlci9wYXRpZW50LndyaXRlIiwidXNlci9wcmFjdGl0aW9uZXIucmVhZCIsInVzZXIvcHJhY3RpdGlvbmVyLndyaXRlIiwidXNlci9wcmVzY3JpcHRpb24ucmVhZCIsInVzZXIvcHJvY2VkdXJlLnJlYWQiLCJ1c2VyL3NvYXBfbm90ZS5yZWFkIiwidXNlci9zb2FwX25vdGUud3JpdGUiLCJ1c2VyL3N1cmdlcnkucmVhZCIsInVzZXIvc3VyZ2VyeS53cml0ZSIsInVzZXIvdHJhbnNhY3Rpb24ucmVhZCIsInVzZXIvdHJhbnNhY3Rpb24ud3JpdGUiLCJ1c2VyL3ZpdGFsLnJlYWQiLCJ1c2VyL3ZpdGFsLndyaXRlIiwidXNlci9BbGxlcmd5SW50b2xlcmFuY2UucmVhZCIsInVzZXIvQ2FyZVRlYW0ucmVhZCIsInVzZXIvQ29uZGl0aW9uLnJlYWQiLCJ1c2VyL0NvdmVyYWdlLnJlYWQiLCJ1c2VyL0VuY291bnRlci5yZWFkIiwidXNlci9JbW11bml6YXRpb24ucmVhZCIsInVzZXIvTG9jYXRpb24ucmVhZCIsInVzZXIvTWVkaWNhdGlvbi5yZWFkIiwidXNlci9NZWRpY2F0aW9uUmVxdWVzdC5yZWFkIiwidXNlci9PYnNlcnZhdGlvbi5yZWFkIiwidXNlci9Pcmdhbml6YXRpb24ucmVhZCIsInVzZXIvT3JnYW5pemF0aW9uLndyaXRlIiwidXNlci9QYXRpZW50LnJlYWQiLCJ1c2VyL1BhdGllbnQud3JpdGUiLCJ1c2VyL1ByYWN0aXRpb25lci5yZWFkIiwidXNlci9QcmFjdGl0aW9uZXIud3JpdGUiLCJ1c2VyL1ByYWN0aXRpb25lclJvbGUucmVhZCIsInVzZXIvUHJvY2VkdXJlLnJlYWQiLCJwYXRpZW50L2VuY291bnRlci5yZWFkIiwicGF0aWVudC9wYXRpZW50LnJlYWQiLCJwYXRpZW50L0FsbGVyZ3lJbnRvbGVyYW5jZS5yZWFkIiwicGF0aWVudC9DYXJlVGVhbS5yZWFkIiwicGF0aWVudC9Db25kaXRpb24ucmVhZCIsInBhdGllbnQvQ292ZXJhZ2UucmVhZCIsInBhdGllbnQvRW5jb3VudGVyLnJlYWQiLCJwYXRpZW50L0ltbXVuaXphdGlvbi5yZWFkIiwicGF0aWVudC9NZWRpY2F0aW9uUmVxdWVzdC5yZWFkIiwicGF0aWVudC9PYnNlcnZhdGlvbi5yZWFkIiwicGF0aWVudC9QYXRpZW50LnJlYWQiLCJwYXRpZW50L1Byb2NlZHVyZS5yZWFkIiwic2l0ZTpkZWZhdWx0Il19.W26UIyF8K3w1yptANugEWZ3qN3qK64z0ikpt0AmX6Ruar43QC_7ah8EJdSY9ualC0_7msThWSKw7LSW1g2b7X1F2LuSJ8aUFrMBp3Fs3fmAuhVM8355AIgkUrCRE6Q24YCNFtEDZynDpPHd30t2Lyt1G3B2cAWCF2nJUaVsjzUotT2Bkv6_J87dq8p3GCNXUnc-SC7rAl6kwwsxq5xTznLysLOBlrdb6d_0l1YNwPnU5Zb0v3hxYX6pETr-8geaOb5GcsdEym5lLFFph4Y_dTEdlFqy-R0lxtL4j4FAO8Wk2kdB_vW9gqqE_4FdOQOKd1A6f3kJguZ3aNk1gyJ-sCw"

df_patient, patient_uuids = fetch_patient_data(api_url, bearer_token)

if df_patient is not None and patient_uuids is not None:
    predictions = log_reg.predict(df_patient[['chest_pain_type', 'max_heart_rate_achieved']].values)  # Generate predictions
    
    # Convert predictions to a list of integers (for JSON serialization)
    predictions = predictions.tolist()

    # Insert predictions back into the database
    insert_prediction(api_url, bearer_token, df_patient, patient_uuids, predictions)
else:
    print("Failed to fetch patient data for prediction.")
