import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data and merge the training files into a single training file

df_train = pd.read_csv('train_values.csv')
df_label = pd.read_csv('Train_Labels.csv')

# Concat an output label from "train_labels" to "train_values" file.

temp = pd.concat([df_train, df_label['heart_disease_present']], axis=1)
df_train = temp

# Change from string to number a "thal" variable in training file.
df_train['thal']=df_train['thal'].map({'normal':0,'reversible_defect':1,'fixed_defect':2})

# Change from string to number a "thal" variable in test file.
df_test=pd.read_csv('test_values.csv')
df_test['thal']=df_test['thal'].map({'normal':0,'reversible_defect':1,'fixed_defect':2})
pd.DataFrame(df_test).to_csv('test.csv', index=False)
print(df_test.shape)

# Generate a processed training file
pd.DataFrame(df_train).to_csv('training.csv', index=False)
print(df_train.shape)
x = df_train.iloc[:, 1:14]
#print(x)

thal_dummies=(pd.get_dummies(x.values[:,1]))
thal_dummies_temp = pd.DataFrame(thal_dummies)

chest_dummies=(pd.get_dummies(x.values[:,3]))
chest_dummies_temp = pd.DataFrame(chest_dummies)

sex_dummies=(pd.get_dummies(x.values[:,9]))
sex_dummies_temp = pd.DataFrame(sex_dummies)

#angina_dummies=(pd.get_dummies(x.values[:,12]))
#print(sex_dummies)
#angina_dummies_temp = pd.DataFrame(angina_dummies)
#print(vessels_dummies_temp)


#print(df_train)
z = df_test.iloc[:,1:14]
thal_dummies=(pd.get_dummies(z.values[:,1]))
thal_dummies_temp = pd.DataFrame(thal_dummies)

chest_dummies=(pd.get_dummies(z.values[:,3]))
chest_dummies_temp = pd.DataFrame(chest_dummies)

sex_dummies=(pd.get_dummies(z.values[:,9]))
sex_dummies_temp = pd.DataFrame(sex_dummies)

#angina_dummies=(pd.get_dummies(z.values[:,12]))
#print(sex_dummies)
#angina_dummies_temp = pd.DataFrame(angina_dummies)
#print(vessels_dummies_temp)


#print(df_test)
#Create a feature Vector "X" and label Vector "y"
# = df_train.iloc[:,1:-1].values

X = df_train[['chest_pain_type', 'max_heart_rate_achieved']].values  # Only use these two features

y = df_train.iloc[:, -1].values
print(y.shape)
print(y)

z = df_test.iloc[:,1:20].values
print(z.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(penalty='l2', class_weight='balanced', random_state=0, solver='lbfgs', 
                             tol=0.00001, C=1.0, fit_intercept=True, max_iter=30000000)
log_reg.fit(X_train, y_train)

# Decision Boundary Visualization
plt.figure(figsize=(10,6))
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='k')
plt.xlabel("Chest Pain Type (Scaled)")
plt.ylabel("Max Heart Rate Achieved (Scaled)")
plt.title("Decision Boundary (Logistic Regression)")
plt.show()
