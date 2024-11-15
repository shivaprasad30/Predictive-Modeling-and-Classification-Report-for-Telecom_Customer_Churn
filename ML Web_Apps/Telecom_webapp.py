import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.header('Interactive Analysis and Prediction of Telecom Customer Churn')

# Load and display data
st.subheader('Telco data')
df = pd.read_csv('Telco_Customer_Churn.csv')
st.dataframe(df.head())

# Data visualization
nav = st.sidebar.radio("Select the Column", ['gender', 'PaymentMethod', 'InternetService'])

f1 = plt.figure(figsize=(4,4))
sns.countplot(x='Churn', hue=nav, data=df)

st.pyplot(f1)

# Data preprocessing
df1 = df[['gender', 'tenure', 'MonthlyCharges', 'PaymentMethod', 'Churn']].copy()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() # LabelEncoder converts Data object to Integrals
df1['gender'] = lb.fit_transform(df1['gender'])
df1['PaymentMethod'] = lb.fit_transform(df1['PaymentMethod'])

df1['Churn'] = lb.fit_transform(df1['Churn'])
st.subheader('Data after LabelEncoding')
st.dataframe(df1.head())

# Split data into features and target
x = df1.iloc[:, :-1]
y = df1.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Model selection
classifier_name = st.sidebar.selectbox('Select Classifier', ['KNN', 'Random Forest', 'Decision Tree'])
def select_params(cls_name):
    params = {}
    if cls_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif cls_name == 'Random Forest':
        n_estimators = st.sidebar.slider('N_estimators', 10, 100)
        criterion = st.sidebar.selectbox('Criteria', ('gini', 'entropy'))
        max_depth = st.sidebar.slider('Max_Depth', 3, 15)
        params['n_estimators'] = n_estimators
        params['criterion'] = criterion
        params['max_depth'] = max_depth
    elif cls_name == 'Decision Tree':
        criterion = st.sidebar.selectbox('Criteria', ('gini', 'entropy'))
        max_depth = st.sidebar.slider('Max_Depth', 3, 15)
        params['criterion'] = criterion
        params['max_depth'] = max_depth
    return params
params = select_params(classifier_name)

# Instantiate and train model
def get_classifier(cls_name, params):
    cls = None
    if cls_name == 'KNN':
        cls = KNeighborsClassifier(n_neighbors=params['K'])
    elif cls_name == 'Random Forest':
        cls = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'], max_depth=params['max_depth'])
    else:
        cls = DecisionTreeClassifier(criterion=params['criterion'], max_depth=params['max_depth'])
    return cls
model = get_classifier(classifier_name, params)
model.fit(x_train, y_train)

# Make predictions and evaluate
ypred = model.predict(x_test)
acc = accuracy_score(y_test, ypred)
cm = confusion_matrix(y_test, ypred)
st.write('Accuracy', acc)
st.write('Confusion Matrix', cm)
st.write('Classification_report', classification_report(y_test, ypred))

# 1)To run Streamlit web app in Browser, open the terminal and write the following
#  streamlit run app.py
# 2)TO stop the running server, in the terminal
#  Press Ctrl + c