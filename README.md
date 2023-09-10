# Cardiovascular_Risk_Prediction

## Table of Contents

- [Project Description](#overview)
- [Dataset](#data)
- [ML Model](#implemented_models)
- [Contributing](#contributing)


## Overview

This project aims to develop a predictive model for assessing the risk of cardiovascular diseases in individuals based on their health attributes. Cardiovascular diseases are a leading cause of mortality worldwide, and early detection and risk assessment can lead to timely interventions and improved healthcare outcomes. We utilize machine learning techniques to analyze medical data and predict the likelihood of an individual developing cardiovascular diseases, providing valuable insights for healthcare professionals and patients.

## Dataset

- **Description:** The dataset contains information related to various health attributes and risk factors associated with cardiovascular diseases. It includes features such as age, gender, blood pressure, cholesterol levels, smoking status, and more.

- **Exploratory Data Analysis (EDA):** 

- **Data Preprocessing:** We perform data preprocessing steps, which includes data cleaning, handling missing values, feature engineering, and data normalization. These steps ensure the dataset is suitable for machine learning model training.

- **Data Splitting:** To train and evaluate our machine learning models, we split the dataset into training and testing subsets.

- **Handling Imbalanced Datasets:** To address the issue of class imbalance (where one class significantly outnumbers the other), we use the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE helps balance the distribution of the target variable by generating synthetic samples for the minority class.

## ML Model

In this project, we have explored several machine learning models to predict cardiovascular risk. Here are the models we have implemented:

- **Logistic Regression:**
  - Description: Logistic Regression is a linear classification algorithm used to model the probability of a binary outcome. It's interpretable and provides insights into feature importance.
  - Implementation: We trained a Logistic Regression model to predict cardiovascular risk based on the selected features.
  - Validation Metrics:
    - Accuracy: 
    - Precision: 
    - Recall: 
    - F1 Score: 
    - ROC AUC: 

- **Decision Tree Classifier:**
  - Description: Decision Tree Classifier is a non-linear algorithm that makes predictions by partitioning the feature space into regions. It's capable of capturing complex relationships in the data.
  - Implementation: We employed a Decision Tree Classifier to analyze the dataset and make predictions.
  - Validation Metrics:
    - Accuracy: 
    - Precision: 
    - Recall: 
    - F1 Score: 
    - ROC AUC: 

- **Support Vector Classifier:**
  - Description: Support Vector Classifier (SVC) is a powerful algorithm for both classification and regression tasks. It finds the optimal hyperplane to separate data points.
  - Implementation: We utilized an SVC to classify individuals into risk categories based on health attributes.
  - Validation Metrics:
    - Accuracy: 
    - Precision: 
    - Recall: 
    - F1 Score: 
    - ROC AUC: 

- **Random Forest Classifier:**
  - Description: Random Forest Classifier is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and reduce overfitting.
  - Implementation: We built a Random Forest Classifier to enhance the predictive performance of our model.
  - Validation Metrics:
    - Accuracy: 
    - Precision: 
    - Recall: 
    - F1 Score: 
    - ROC AUC: 

- **HistGradientBoostingClassifier:**
  - Description: HistGradientBoostingClassifier is a boosting algorithm that optimizes the prediction by combining the results of multiple weak learners (typically decision trees).
  - Implementation: We employed HistGradientBoostingClassifier to create an ensemble model for cardiovascular risk prediction.
  - Validation Metrics:
    - Accuracy: 
    - Precision: 
    - Recall: 
    - F1 Score: 
    - ROC AUC: 

Each of these models was trained, evaluated, and fine-tuned to provide the best possible predictions for cardiovascular risk assessment. Validation metrics such as accuracy, precision, recall, F1 score, and ROC AUC demonstrate the performance of each model on the validation dataset.
