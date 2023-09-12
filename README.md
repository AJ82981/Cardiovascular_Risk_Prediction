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

In this project, we have explored several machine learning models to predict cardiovascular risk. Here are the models we have implemented along with their descriptions and validation metrics:

### Logistic Regression:

**Description:** Logistic Regression is a linear classification algorithm used to model the probability of a binary outcome. It's interpretable and provides insights into feature importance.

**Implementation:** We trained a Logistic Regression model to predict cardiovascular risk based on the selected features.

**Validation Metrics:**
- Accuracy: 0.70
- Precision: 0.28
- Recall: 0.66
- F1 Score: 0.40

### Random Forest Classifier:

**Description:** Random Forest Classifier is an ensemble learning method that combines multiple decision trees to improve predictive accuracy and reduce overfitting.

**Implementation:** We built a Random Forest Classifier to enhance the predictive performance of our model.

**Validation Metrics:**
- Accuracy: 0.82
- Precision: 0.33
- Recall: 0.19
- F1 Score: 0.24

### Support Vector Classifier (SVC):

**Description:** Support Vector Classifier (SVC) is a powerful algorithm for both classification and regression tasks. It finds the optimal hyperplane to separate data points.

**Implementation:** We utilized an SVC to classify individuals into risk categories based on health attributes.

**Validation Metrics:**
- Accuracy: 0.85
- Precision: 0.00
- Recall: 0.00
- F1 Score: 0.00

### Decision Tree Classifier:

**Description:** Decision Tree Classifier is a non-linear algorithm that makes predictions by partitioning the feature space into regions. It's capable of capturing complex relationships in the data.

**Implementation:** We employed a Decision Tree Classifier to analyze the dataset and make predictions.

**Validation Metrics:**
- Accuracy: 0.80
- Precision: 0.29
- Recall: 0.24
- F1 Score: 0.26

### Histogram-Based Gradient Boosting Classifier (HistGradientBoostingClassifier):

**Description:** HistGradientBoostingClassifier is a boosting algorithm that optimizes the prediction by combining the results of multiple weak learners (typically decision trees).

**Implementation:** We used the HistGradientBoostingClassifier to create an ensemble model for cardiovascular risk prediction.

**Validation Metrics:**
- Accuracy: 0.81
- Precision: 0.20
- Recall: 0.10
- F1 Score: 0.13

### Gradient Boosting Classifier:

**Description:** Gradient Boosting is a machine learning technique for regression and classification problems that builds an additive model in a forward stage-wise manner.

**Implementation:** We applied Gradient Boosting to improve the model's predictive performance.

**Validation Metrics:**
- Accuracy: 0.78
- Precision: 0.26
- Recall: 0.25
- F1 Score: 0.25

Each of these models was trained, evaluated, and fine-tuned to provide the best possible predictions for cardiovascular risk assessment. The validation metrics mentioned above demonstrate the performance of each model on the validation dataset.
