import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score,classification_report

df=pd.read_csv("D:\git_hub\ML-Projects\scripts\insurance_data.csv")
df.head()

x=df.drop(columns="bought_insurance")
y=df["bought_insurance"]

sc=StandardScaler()

x_scaled=sc.fit_transform(x)

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,train_size=0.8)

reg=LogisticRegression()

reg.fit(x_train,y_train)

y_predicted=reg.predict(x_test)
y_predicted

accuracy=accuracy_score(y_test,y_predicted)
accuracy

print(f"The accuracy is {accuracy}")

print(classification_report(y_test,y_predicted,target_names=["Yes","No"]))

