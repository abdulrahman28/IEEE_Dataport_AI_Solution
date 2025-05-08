# IEEE_Dataport_AI_Solution
 IEEE Dataport AI Solution (Wearable Health Hackathon - Develop a solution to early diagnose diseases using non-invasive methods and health data)

 # This work is on Development of a robust model for the detection of coronary heart disease using machine learning techniques 

# Problem the work is addressing:
* According to the Global Burden of Disease (GBD) study in 2013, the chronic disease group has a high mortality rate and is frequently found in high-income countries such as the United States, which has 87% of deaths caused by chronic diseases. 
* The conventional method of diagnosis of coronary heart disease (CHD) is very expensive, with different side effects, and requires strong technological knowledge.
* This study proposed the development of a robust model for the detection of coronary heart disease using machine learning (ML) techniques.


# Aim and Objectives of the work:
The work aims to develop a robust model that can detect coronary heart disease using machine learning techniques.
The objectives are to:
* Process and clean the coronary heart disease public dataset.
* Compute and interpret the correlations of all the features.
* Apply validation set cross-validation technique to train and test different ML models to select the best ML model.
* Apply the k-fold cross-validation technique to tune the best model hyperparameters for optimal performance.
* Evaluate the performance of the model.

# Significance of the work:
* Low-cost and non-invasive approach for the diagnosis of coronary heart disease (CHD).
* Early detection of coronary heart disease for quick treatment.
* Aid medical personnel in the quick diagnosis of CHD.
* Integrated into a wearable heart monitoring device for a patient with a hereditary trait of CHD.


# CHD dataset information:
Public Dataset: CVD_FinalData.csv
Number of samples: 5390
Predictors: 21
Numerical data: 12 (Age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartrate, glucose, Triglycerdie, hdl_cholesterol, ldl_cholesterol, CPK_MB_Percentage)
Categorical Data: 9 
Binary category: 7 (sex, is_smoking, BPMeds, prevalentStroke, prevalentHyp, diabetes, exng)
Ordinal category: 2 (education, caa)
Response or Target Variable: TenYearCHD (0 or 1) (Binary Classification)

# ML Classifiers considered:
* k-nearest neigbors (k-NN)
* Support vector machine (SVM)
* Decision tree (DT)
* Random forest (RF)
* Logistic regression (LR)
* Gaussian naïve bayes (GNB)
* Linear discriminant analysis (LDA)
* Light gradient boost
* Extreme gradient boost (XGBoost)
* CatBoost

# Validation set cross-validation technique (80%-20%) was adopted to select the model

The random forest has the best performance with test accuracy, sensitivity, specificity, precision, F1-score, and G-mean of 97.41%, 97.42%, 97.42%, 97.54%, 97.48%, and 97.42%, respectively.

# K-fold cross-validation technique (80%-20%) was adopted to tune the hyperparameters (number of trees and max_depth) of the random forest model

Number of trees: 300, Max_depth: None

The result of the hyperparameter tuning: test accuracy, sensitivity, specificity, precision, F1-score, and G-mean of 97.71%, 97.71%, 97.81%, 97.54%, 97.76%, and 97.71%, respectively. The AUROC of the random forest model is 1.00.