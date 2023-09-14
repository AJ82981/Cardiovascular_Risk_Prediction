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


# ML Model Comparison and Evaluation

In this section, we evaluate and compare the performance of different machine learning models for predicting cardiovascular risk based on selected features. We examine several classification models, including Logistic Regression, Decision Tree Classifier, Histogram-Based Gradient Boosting (HGB) Classifier, Random Forest Classifier, and Support Vector Classifier (SVC). For each model, we provide a brief description, implementation details, and validation metrics for both the original dataset and a dataset resampled using SMOTE.

## Logistic Regression:
**Description:** Logistic Regression is a linear classification algorithm used to model the probability of a binary outcome. It's interpretable and provides insights into feature importance.

**Implementation:** We trained a Logistic Regression model to predict cardiovascular risk based on the selected features.

**Validation Metrics (SMOTE X_train, y_train):**
- Accuracy: 0.70
- Precision: 0.28
- Recall: 0.66
- F1 Score: 0.40

## Decision Tree Classifier:
**Description:** The Decision Tree Classifier is a non-linear classification algorithm that recursively splits data into subsets based on the most significant attribute, resulting in a tree-like structure.

**Implementation:** We trained a Decision Tree Classifier model to predict cardiovascular risk based on the selected features.

**Validation Metrics (X_train, y_train):**
- Accuracy: 0.75
- Precision: 0.22
- Recall: 0.25
- F1 Score: 0.24

**Validation Metrics (SMOTE X_train, y_train):**
- Accuracy: 0.71
- Precision: 0.15
- Recall: 0.20
- F1 Score: 0.17

## Histogram-Based Gradient Boosting Classifier (HGB):
**Description:** The Histogram-Based Gradient Boosting Classifier is an ensemble learning method that builds an additive model of decision trees to predict outcomes.

**Implementation:** We trained an HGB model to predict cardiovascular risk based on the selected features.

**Validation Metrics (X_train, y_train):**
- Accuracy: 0.83
- Precision: 0.30
- Recall: 0.08
- F1 Score: 0.12

**Validation Metrics (SMOTE X_train, y_train):**
- Accuracy: 0.82
- Precision: 0.26
- Recall: 0.10
- F1 Score: 0.14

## Random Forest Classifier:
**Description:** The Random Forest Classifier is an ensemble learning method that builds multiple decision trees and combines their predictions for improved accuracy and generalization.

**Implementation:** We trained a Random Forest Classifier model to predict cardiovascular risk based on the selected features.

**Validation Metrics (SMOTE X_train, y_train):**
- Accuracy: 0.82
- Precision: 0.31
- Recall: 0.17
- F1 Score: 0.22

**Validation Metrics (Unsampled X_train, y_train):**
- Accuracy: 0.83
- Precision: 0.37
- Recall: 0.19
- F1 Score: 0.25

## Gradient Boosting Classifier:
**Description:** The Gradient Boosting Classifier is an ensemble learning method that builds an additive model of decision trees to predict outcomes.

**Implementation:** We trained a Gradient Boosting Classifier model to predict cardiovascular risk based on the selected features.

**Validation Metrics (SMOTE X_train, y_train):**
- Accuracy: 0.78
- Precision: 0.26
- Recall: 0.25
- F1 Score: 0.25

**Validation Metrics After Hyperparameter Tuning (SMOTE X_train, y_train):**
- Accuracy: 0.80
- Precision: 0.17
- Recall: 0.08
- F1 Score: 0.11

## Support Vector Classifier:
**Description:** The Support Vector Classifier (SVC) is a linear classifier that aims to find the hyperplane that best separates the classes.

**Implementation:** We trained a Support Vector Classifier model to predict cardiovascular risk based on the selected features.

**Validation Metrics (X_train, y_train):**
- Accuracy: 0.85
- Precision: 0.00
- Recall: 0.00
- F1 Score: 0.00

## Summary:
Among the tested models, the Random Forest Classifier with unsampled X_train, y_train data achieved the highest F1 score (0.25) and a relatively balanced precision and recall.

The Logistic Regression model with SMOTE X_train, y_train data achieved the highest F1 score (0.40) but has a relatively low precision and a high recall for class 1.0, indicating a trade-off between precision and recall.

The Decision Tree Classifier showed a balance between precision and recall in the X_train, y_train scenario.

The Histogram-Based Gradient Boosting Classifier (HGB) did not perform as well in terms of F1 score or recall in both scenarios.

The Support Vector Classifier did not perform well in terms of classification metrics, achieving an F1 score of 0.00.

