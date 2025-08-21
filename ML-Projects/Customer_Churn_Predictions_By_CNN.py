import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns="customerID", inplace=True)

df[pd.to_numeric(df["TotalCharges"], errors="coerce").isnull()]
df1 = df[df.TotalCharges != " "]
df1["TotalCharges"] = pd.to_numeric(df1.TotalCharges)

tenure_churn_no = df1[df1.Churn == "No"].tenure
tenure_churn_yes = df1[df1.Churn == "Yes"].tenure

plt.hist([tenure_churn_no, tenure_churn_yes], color=["red", "blue"])

def print_data(df):
    for col in df:
        if df[col].dtypes == "object":
            print(f"{col}, {df[col].unique()}")

print_data(df1)

df1.replace("No phone service", "No", inplace=True)
df1.replace("No internet service", "No", inplace=True)

yes_or_no = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]

for col in yes_or_no:
    df1[col].replace({"Yes": 1, "No": 0}, inplace=True)

df1["gender"] = df1["gender"].replace({"Female": 0, "Male": 1})

df2 = pd.get_dummies(df1, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

scaler = MinMaxScaler()
df2[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(df2[["tenure", "MonthlyCharges", "TotalCharges"]])

x = df2.drop(columns="Churn")
y = df2["Churn"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

model = keras.Sequential([
    keras.layers.Dense(23, input_shape=(23,), activation="sigmoid"),
    keras.layers.Dense(25, activation="sigmoid"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

y_predicted = model.predict(x_test)

y_predicted_prob = [1 if i > 0.5 else 0 for i in y_predicted]
