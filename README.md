# **HEART DISEASE PREDICTION USING PATIENT DATA**
### **Logistic regression**

---
## Notebook used in Data Analysis and Model Building
[heart_disease_prediction.ipynb]( )
## **Dataset Link**
[heart_disease.csv](https://github.com/AbelEsther/HEART-DISEASE-PREDICTION/blob/c1d918f95e54f039267226c028cd96c452c92df0/heart_disease.csv)

##  Problem Introduction & Motivation

Heart disease is one of the leading causes of mortality worldwide, accounting for a significant portion of global deaths each year. Early detection and intervention can greatly improve patient outcomes, reducing the risk of severe complications or death. However, diagnosing heart disease is complex, as it often requires analyzing multiple physiological indicators and symptoms that may not always clearly indicate the presence of the disease.

In this project, I aim to develop a predictive model that uses key patient characteristics to identify the likelihood of heart disease. By utilizing logistic regression, a statistical technique well-suited for binary classification tasks, I seek to build a model that can classify patients into two categories: those likely to have heart disease and those unlikely to have it. This model will serve as an initial screening tool to assist healthcare professionals in identifying high-risk patients, allowing for timely and potentially life-saving medical intervention.

### Dataset Description

- **Source**: Collected data from approximately 300 medical patients evaluated for heart disease.
- **Target Variable**: Binary outcome indicating heart disease presence (1) or absence (0).
- **Predictor Variables**: Five patient characteristics (to be defined upon data inspection).

The Heart Disease Dataset contains information on nearly 300 medical patients evaluated for the presence of heart disease. Each record provides details on different patient characteristics that may influence the likelihood of a heart disease diagnosis. The dataset includes the following attributes:

### Description of variables
- **age**: Age of the patient in years (e.g., 52)
- **sex**: Gender of the patient (1 = male; 0 = female) (e.g., 1)
- **max_heart_rate**: Maximum heart rate (e.g., 160)
- **angina_level**: Severity level of angina experienced by patient(e.g., 'no', 'mild', 'severe') - a categorical variable used to gauge angina severity
- **non_angina_rate**: Rate of non-angina related symptoms (e.g., 0.5) - provides a measure of symptom rate not directly related to angina
- **diagnosis**: Presence of heart disease (1 = disease; 0 = no disease) - This is the target variable.

I will use this dataset for a classification analysis, focusing on predicting the `diagnosis` (presence of heart disease) based on the available patient characteristics.

---
### Data Preparation (EDA)
*   Loading the Data
*   Data Cleaning
*   Handling Missing Data
*   Transforming
*   Summarizing and Visualization
---
### Loading the data
*   Load the dataset by importing the neccesary libraries and reading the csv file from google drive
*   Load the uploaded CSV file into a DataFrame
*   Inspect its structure, types, and basic statistics to gain initial insights.

---
## Model Evaluation Summary

Based on the displayed results, here’s an evaluation of the model’s performance:

### Model Evaluation Summary

1. **Accuracy**:
   - **74.67%**: The model correctly predicted heart disease status 74.67% of the time, which indicates moderate overall performance. However, this accuracy might not be sufficient for a healthcare application where both false positives and false negatives can have serious implications.

2. **Confusion Matrix**:
   - **True Negatives (TN)**: 34 (correctly predicted as "No heart disease")
   - **False Positives (FP)**: 8 (incorrectly predicted as "Heart disease" when they don’t have it)
   - **True Positives (TP)**: 22 (correctly predicted as "Heart disease")
   - **False Negatives (FN)**: 11 (incorrectly predicted as "No heart disease" when they actually have it)
   - This confusion matrix shows that the model has a relatively balanced distribution of errors, but it misses 11 true cases of heart disease, which is concerning in a healthcare context.

3. **Precision, Recall, and F1-Score**:
   - **Class 0 (No Heart Disease)**:
     - **Precision**: 75.56% — When the model predicts "No heart disease," it is correct about 75.56% of the time.
     - **Recall**: 80.95% — The model correctly identifies 80.95% of patients who do not have heart disease.
     - **F1-Score**: 78.16% — Indicates a reasonable balance between precision and recall for this class.
   - **Class 1 (Heart Disease)**:
     - **Precision**: 73.33% — When the model predicts "Heart disease," it is correct about 73.33% of the time.
     - **Recall**: 66.67% — The model correctly identifies 66.67% of the patients who actually have heart disease, missing 33.33%.
     - **F1-Score**: 69.84% — Shows moderate performance in identifying heart disease cases but suggests there is room for improvement.
   - **Macro Avg and Weighted Avg**:
     - The macro and weighted average F1-scores are around **74%**, suggesting the model is reasonably balanced in its performance across both classes but has some difficulty in capturing true positives for heart disease cases.

4. **Model Coefficients**:
   - The coefficients provide insight into how each feature impacts the model’s predictions:
     - **age**: Positive coefficient (0.035898) suggests a slight increase in the likelihood of heart disease with age.
     - **sex_LabelEncoded**: Positive coefficient (1.074011) indicates that being male (assuming 1 represents male) is associated with a higher risk of heart disease.
     - **max_heart_rate**: Negative coefficient (-0.025704) suggests a small decrease in heart disease likelihood as max heart rate increases.
     - **angina_level_OrdinalEncoded**: Positive coefficient (1.043699) implies that higher angina levels increase the likelihood of heart disease, which aligns with medical expectations.
     - **non_anginal_pain**: Negative coefficient (-0.900047) suggests that higher non-anginal pain is associated with a decreased likelihood of heart disease, as non-anginal symptoms may be less related to heart disease.

### Summary of Insights
- **Moderate Accuracy**: An accuracy of 74.67% indicates that the model performs fairly well but could benefit from improvements, particularly in identifying true cases of heart disease.
- **Precision and Recall Balance**: Precision is moderately high for both classes, but recall for the heart disease class (66.67%) is relatively low, meaning the model may miss a significant portion of actual heart disease cases.
- **Coefficient Interpretations**: The coefficients align with medical knowledge, with age, male sex, and angina level increasing heart disease risk. However, the impact of `non_anginal_pain` as a negative predictor might need further analysis to ensure it's capturing meaningful patterns.

---

## **Analysis Summary**

In this analysis, I aimed to build a predictive model to classify patients as having heart disease or not based on key medical and demographic characteristics. My workflow included data preparation, model selection/building, and model evaluation using logistic regression. Below is a summary of each step:

#### **1. Data Cleaning and Transformation**
   - **Handling Missing Values**: I checked for missing values in the dataset, removing rows with insufficient data to maintain model integrity.
   - **Feature Encoding**: For categorical features like `sex` and `angina_level`, I used **Label Encoding** and **Ordinal Encoding** to convert these text-based categories into numeric representations, making them suitable for logistic regression.
   - **Outlier Treatment**: Negative values in features such as `max_heart_rate` were identified and converted to their positive counterparts as they were suspected to be entry errors. This ensured logical consistency across the dataset.
   

#### **2. Model Building**
   - **Feature Selection**: I selected five key features (`age`, `sex_LabelEncoded`, `max_heart_rate`, `angina_level_encoded`, and `non_anginal_pain`) based on their potential relevance to heart disease.
   - **Train-Test Split**: The data was split into training and testing sets, with 70% of the data used for training the model and 30% for testing. This provided an unbiased evaluation of the model’s performance.
   - **Logistic Regression**: I chose logistic regression as my model due to its effectiveness in binary classification problems and its interpretability in healthcare applications.

#### **3. Model Evaluation**
   - **Accuracy**: The model achieved an accuracy of approximately 74.67%, indicating that it correctly classified heart disease status in most cases.
   - **Confusion Matrix**: I used a confusion matrix to visualize the distribution of true positives, true negatives, false positives, and false negatives. This helped in understanding the types of errors the model makes.
   - **Precision, Recall, and F1-Score**: Precision and recall were calculated for both classes (heart disease and no heart disease) to measure the model’s ability to correctly identify positive cases while minimizing false positives. The F1-score provided a balance between precision and recall.
   - **ROC-AUC**: ROC-AUC was calculated to measure the model's ability to discriminate between the classes across various thresholds. This metric is especially valuable in assessing model performance in scenarios with imbalanced data.

#### **4. Coefficient Interpretation**
   - The coefficients from the logistic regression model were analyzed to understand the impact of each feature on heart disease risk. Features like `age` and `sex_LabelEncoded` had positive coefficients, indicating a higher likelihood of heart disease with increased age or being male. Conversely, `non_anginal_pain` had a negative coefficient, suggesting it was less associated with heart disease.

### **Conclusion**
This analysis provided a comprehensive workflow for building a heart disease classification model using logistic regression. The model demonstrated moderate accuracy and balance between precision and recall.

<br>**Future improvements** could focus on optimizing recall to reduce false negatives and exploring additional features or alternative models to enhance predictive power.

