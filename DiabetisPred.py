import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("/content/diabetes.csv")
print(data)

sns.heatmap(data.isnull())

sns.countplot(data=data,x="Outcome")

data[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]]=data[["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]].replace(0,np.NaN)

sns.heatmap(data.isnull())

data['Glucose'].fillna(data['Glucose'].mean(),inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(),inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].mean(),inplace=True)
data['Insulin'].fillna(data['Insulin'].mean(),inplace=True)
data['BMI'].fillna(data['BMI'].mean(),inplace=True)

sns.heatmap(data.isnull())

print(data)

data=data.drop(["Pregnancies","DiabetesPedigreeFunction","Age"],axis=1)

print(data)

sns.pairplot(data,hue="Outcome")

x=data.iloc[:,:-1].values
y=data.iloc[:,-1]
print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

from sklearn.svm import SVC
svm_model=SVC(kernel="linear")
svm_model.fit(x_train,y_train)

yp=svm_model.predict(x_test)

yp=svm_model.predict(x_test)
from sklearn.metrics import classification_report
cr=classification_report(y_test,yp)
print(cr)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

yp=svm_model.predict(x_test)
from sklearn.metrics import classification_report
cr=classification_report(y_test,yp)
print(cr)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

yp=svm_model.predict(x_test)
from sklearn.metrics import classification_report
cr=classification_report(y_test,yp)
print(cr)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)

yp=svm_model.predict(x_test)
from sklearn.metrics import classification_report
cr=classification_report(y_test,yp)
print(cr)