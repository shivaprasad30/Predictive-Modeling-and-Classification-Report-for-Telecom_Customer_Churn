# Predictive-Modeling-and-Classification-Report-for-Telecom-Customer-Churn
## Overview
This project aims to predict customer churn in the telecom industry using machine learning models. The dataset used is the Telco Customer Churn dataset, and the project is implemented using Python with a Streamlit web app for interactive data visualization and model prediction.
## Dataset
The dataset used in this project is the Telco Customer Churn dataset, which contains information about a telecom company's customers and whether they have churned (i.e., discontinued their service) or not. The dataset includes various features such as: <br>
<ul >
  <li>gender</li>
  <li>tenure</li>
  <li>MonthlyCharges</li>
  <li>PaymentMethod</li>
  <li>Churn (target variable)</li>
</ul>

## Usage
1.Load the dataset and display the first few rows. <br>
2.Use the sidebar to select different columns for data visualization. <br>
3.Preprocess the data by label encoding categorical features. <br>
4.Split the data into training and testing sets. <br>
5.Select a classifier and set its parameters. <br>
6.Train the model and display performance metrics such as accuracy, confusion matrix, and classification report. <br>

## Features
**Interactive Data Visualization:** Use Streamlit to visualize the distribution of various features and their relationship with churn. <br>
**Data Preprocessing:** Handle categorical data using label encoding. <br>
**Model Training:** Train different machine learning models (KNN, Random Forest, Decision Tree) to predict customer churn. <br>
**Model Evaluation:** Evaluate the model's performance using accuracy, confusion matrix, and classification report. <br>

## Model Performance
The model's performance is evaluated using several metrics:

**Accuracy:** 75.35% <br>
**Confusion Matrix:**
<ul>
  <li><b>True Negatives (TN):</b> 1112 (The number of non-churn customers correctly identified as non-churn)</li>
  <li><b>False Positives (FP):</b> 186 (The number of non-churn customers incorrectly identified as churn)</li>
  <li><b>False Negatives (FN):</b> 251 (The number of churn customers incorrectly identified as non-churn)</li>
  <li><b>True Positives (TP):</b> 212 (The number of churn customers correctly identified as churn)</li>
</ul>

**Classification Report:**
Precision, Recall, and F1-Score for both churn (1) and non-churn (0) classes.
              precision    recall  f1-score   support

           0       0.82      0.86      0.84      1298
           1       0.54      0.46      0.50       463

    accuracy                           0.75      1761
   macro avg       0.68      0.66      0.67      1761
weighted avg       0.74      0.75      0.75      1761

![image](https://github.com/user-attachments/assets/bf49c455-ecd8-4719-a0e9-5e2e5ff1a999)
