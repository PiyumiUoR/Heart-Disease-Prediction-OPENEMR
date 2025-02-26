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
bearer_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiJQbFZoRDdNT3dYRW1iNXZ1VHFLYUVWRmR0dDRRejdVeDlxZW5JYmF2WGVBIiwianRpIjoiYjg5M2UyOGI0NjRmZDBkZjFiNTQyZDI2NWE2MzczMzk4Zjk1MjMxNTA3ZjdiNTRlZWJjZjg1YTI0YzY4YjYyNmFlZWU5ZjIyN2I4ODAwOGEiLCJpYXQiOjE3NDA1Mjk2NTQuOTIzMTY4LCJuYmYiOjE3NDA1Mjk2NTQuOTIzMTczLCJleHAiOjE3NDA1MzMyNTQuODczNjQzLCJzdWIiOiI5ZTE1YjcwMS1jYzI4LTQzMTYtOTJhYi02ODQ3OTdkZTRiNTAiLCJzY29wZXMiOlsib3BlbmlkIiwib2ZmbGluZV9hY2Nlc3MiLCJhcGk6b2VtciIsImFwaTpmaGlyIiwiYXBpOnBvcnQiLCJ1c2VyL2FsbGVyZ3kucmVhZCIsInVzZXIvYWxsZXJneS53cml0ZSIsInVzZXIvYXBwb2ludG1lbnQucmVhZCIsInVzZXIvYXBwb2ludG1lbnQud3JpdGUiLCJ1c2VyL2RlbnRhbF9pc3N1ZS5yZWFkIiwidXNlci9kZW50YWxfaXNzdWUud3JpdGUiLCJ1c2VyL2RvY3VtZW50LnJlYWQiLCJ1c2VyL2RvY3VtZW50LndyaXRlIiwidXNlci9kcnVnLnJlYWQiLCJ1c2VyL2VuY291bnRlci5yZWFkIiwidXNlci9lbmNvdW50ZXIud3JpdGUiLCJ1c2VyL2ZhY2lsaXR5LnJlYWQiLCJ1c2VyL2ZhY2lsaXR5LndyaXRlIiwidXNlci9pbW11bml6YXRpb24ucmVhZCIsInVzZXIvaW5zdXJhbmNlLnJlYWQiLCJ1c2VyL2luc3VyYW5jZS53cml0ZSIsInVzZXIvaW5zdXJhbmNlX2NvbXBhbnkucmVhZCIsInVzZXIvaW5zdXJhbmNlX2NvbXBhbnkud3JpdGUiLCJ1c2VyL2luc3VyYW5jZV90eXBlLnJlYWQiLCJ1c2VyL2xpc3QucmVhZCIsInVzZXIvbWVkaWNhbF9wcm9ibGVtLnJlYWQiLCJ1c2VyL21lZGljYWxfcHJvYmxlbS53cml0ZSIsInVzZXIvbWVkaWNhdGlvbi5yZWFkIiwidXNlci9tZWRpY2F0aW9uLndyaXRlIiwidXNlci9tZXNzYWdlLndyaXRlIiwidXNlci9wYXRpZW50LnJlYWQiLCJ1c2VyL3BhdGllbnQud3JpdGUiLCJ1c2VyL3ByYWN0aXRpb25lci5yZWFkIiwidXNlci9wcmFjdGl0aW9uZXIud3JpdGUiLCJ1c2VyL3ByZXNjcmlwdGlvbi5yZWFkIiwidXNlci9wcm9jZWR1cmUucmVhZCIsInVzZXIvc29hcF9ub3RlLnJlYWQiLCJ1c2VyL3NvYXBfbm90ZS53cml0ZSIsInVzZXIvc3VyZ2VyeS5yZWFkIiwidXNlci9zdXJnZXJ5LndyaXRlIiwidXNlci90cmFuc2FjdGlvbi5yZWFkIiwidXNlci90cmFuc2FjdGlvbi53cml0ZSIsInVzZXIvdml0YWwucmVhZCIsInVzZXIvdml0YWwud3JpdGUiLCJ1c2VyL0FsbGVyZ3lJbnRvbGVyYW5jZS5yZWFkIiwidXNlci9DYXJlVGVhbS5yZWFkIiwidXNlci9Db25kaXRpb24ucmVhZCIsInVzZXIvQ292ZXJhZ2UucmVhZCIsInVzZXIvRW5jb3VudGVyLnJlYWQiLCJ1c2VyL0ltbXVuaXphdGlvbi5yZWFkIiwidXNlci9Mb2NhdGlvbi5yZWFkIiwidXNlci9NZWRpY2F0aW9uLnJlYWQiLCJ1c2VyL01lZGljYXRpb25SZXF1ZXN0LnJlYWQiLCJ1c2VyL09ic2VydmF0aW9uLnJlYWQiLCJ1c2VyL09yZ2FuaXphdGlvbi5yZWFkIiwidXNlci9Pcmdhbml6YXRpb24ud3JpdGUiLCJ1c2VyL1BhdGllbnQucmVhZCIsInVzZXIvUGF0aWVudC53cml0ZSIsInVzZXIvUHJhY3RpdGlvbmVyLnJlYWQiLCJ1c2VyL1ByYWN0aXRpb25lci53cml0ZSIsInVzZXIvUHJhY3RpdGlvbmVyUm9sZS5yZWFkIiwidXNlci9Qcm9jZWR1cmUucmVhZCIsInBhdGllbnQvZW5jb3VudGVyLnJlYWQiLCJwYXRpZW50L3BhdGllbnQucmVhZCIsInBhdGllbnQvQWxsZXJneUludG9sZXJhbmNlLnJlYWQiLCJwYXRpZW50L0NhcmVUZWFtLnJlYWQiLCJwYXRpZW50L0NvbmRpdGlvbi5yZWFkIiwicGF0aWVudC9Db3ZlcmFnZS5yZWFkIiwicGF0aWVudC9FbmNvdW50ZXIucmVhZCIsInBhdGllbnQvSW1tdW5pemF0aW9uLnJlYWQiLCJwYXRpZW50L01lZGljYXRpb25SZXF1ZXN0LnJlYWQiLCJwYXRpZW50L09ic2VydmF0aW9uLnJlYWQiLCJwYXRpZW50L1BhdGllbnQucmVhZCIsInBhdGllbnQvUHJvY2VkdXJlLnJlYWQiLCJzaXRlOmRlZmF1bHQiXX0.Z7AGtw2QFj0aitPnlNYkvYksNLLB78QV46xW_lLaObXPGAGK6_c7_u9rMdAy0Sf1q7bdDDXLsNCxu76h0rI1JTiKVuFZOUKfdyhv9L_kN-NCWKexNFrTQLEz-mpl9Kuxxnnjpwt5276-I2_S4DZ7dAN_zQ2v0l_NWcdgWJ0DIQxOPj5Eh19Hhnij0LpOKJisgX0KDgjkM3s1Dq0xfWX30EmcWTwUMvtki4v_pade2OC3NPz-NXKmlglIngzxkuTs96LplIFKM9NdMiwObdLfIEX6BL29WTf_MerkqJz7_NU4jXo6GK1QSkM6fzFmhmDl-1ZdSFv2TjI-s-lt8wxvzw"

df_patient, patient_uuids = fetch_patient_data(api_url, bearer_token)

if df_patient is not None and patient_uuids is not None:
    predictions = log_reg.predict(df_patient[['chest_pain_type', 'max_heart_rate_achieved']].values)  # Generate predictions
    
    # Convert predictions to a list of integers (for JSON serialization)
    predictions = predictions.tolist()

    # Insert predictions back into the database
    insert_prediction(api_url, bearer_token, df_patient, patient_uuids, predictions)
else:
    print("Failed to fetch patient data for prediction.")
