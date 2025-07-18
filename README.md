# Project Description
This project aims to build and evaluate multiple machine learning models based on features including blood routine tests, biochemical indicators to diagnose cardiovascular disease (CVD) and its subtypes.


# Importance
Our study provides a practical diagnostic approach for cardiovascular diseases (CVD) by integrating machine learning models with routine clinical indices. This method enables early identification of CVD in high-risk populations such as individuals with hypertension (HTN), diabetes mellitus (DM), and hyperlipidemia (HCL), which can support timely intervention, reduce disease burden, and improve patient outcomes.


# Requirements

The main requirements are listed below:

*  Python 3.11
*  R 4.5.0
*  joblib 1.3.2
*  numpy  1.26.4
*  pandas 2.2.0
*  scikit-learn  1.4.0
*  seaborn   0.13.2
*  dplyr	1.1.4
*  ggplot2	3.5.1
*  ggridges	0.5.6
*  ggsankey	0.0.99999
*  circlize	0.4.16
*  ggvenn	0.1.10
*  UpSetR	1.4.0

# Code Workflow for CVD Diagnosis Modeling Using Machine Learning

* Feature Selection

  Select features with missing values less than 50% to retain high-quality variables for modeling.

* Hyperparameter Optimization

  To optimize hyperparameters, we employed a combination of grid search and cross-validation (CV), which was further refined through manual fine-tuning.

* Diagnostic Model Construction

  We constructed four machine learning diagnostic models: Support Vector Machine (SVM), Random Forest (RF), Logistic Regression (LR), and Extreme Gradient Boosting (XGBoost). Based on performance comparison, XGBoost, which achieved the best results, was selected for further analysis.

* SHAP-Based Feature Importance Visualization

  We applied SHAP (SHapley Additive exPlanations) to interpret the best-performing model and identified the top 10 most important features. Each of these features was then visualized individually to illustrate its contribution to cardiovascular disease diagnosis.



