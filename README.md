# Employee Turnover Prediction using Machine Learning

## Business Problem
Organizations lose significant resources when employees leave.

This project predicts employee turnover using Logistic Regression,
allowing HR teams to identify employees at risk of leaving and take
proactive retention measures.

## Dataset Features
- Age
- Monthly Income
- Job Satisfaction
- Work Life Balance
- Years At Company
- Distance From Home
- Performance Rating

## Project Workflow
1. Data Cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. Feature Scaling
5. Logistic Regression
6. L1 & L2 Regularization
7. Hyperparameter Tuning
8. Model Evaluation

## Results
L2 regularization penalizes large coefficients and helps reduce overfitting while retaining all features.



                   precision    recall  f1-score   support

           0       0.87      0.83      0.85       125
           1       0.86      0.89      0.87       145

    accuracy                           0.86       270
   macro avg       0.86      0.86      0.86       270
weighted avg       0.86      0.86      0.86       270

## Business Impact
The model helps HR teams identify employees likely to resign,
reducing hiring costs and improving workforce retention.

## Streamlit App
streamlit run app.py

• Developed an Employee Turnover Prediction model using Logistic Regression, achieving high classification performance through L1/L2 regularization and hyperparameter tuning.

• Performed end-to-end machine learning workflow including EDA, feature scaling, model training, evaluation, and business recommendation generation.

• Built HR analytics solution to identify employees at risk of attrition, enabling data-driven retention strategies.
