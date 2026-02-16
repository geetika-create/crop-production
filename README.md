# Crop Production Prediction and Analysis
This project focuses on analyzing and predicting crop production based on various environmental and agricultural factors. It involves data cleaning, statistical analysis, and machine learning modeling using a Random Forest Regressor.

## Project Overview
The goal of this project is to understand the drivers of crop production and build a predictive model to estimate future yields. The analysis includes:
- Data Loading: Importing crop production data from a CSV file.
- Data Imputation: Handling missing values using K-Nearest Neighbors (KNN) regression to impute missing Deforestation, Pesticides, and Irrigation data.
- Statistical Analysis:
  - Descriptive statistics of the dataset.
  - Outlier identification using the Interquartile Range (IQR) method.
  - Distribution analysis of crop production.
  - Correlation analysis between various factors (Nitrogen, Temperature, Carbon Dioxide, Methane, Deforestation, Precipitation, Pesticides, Phosphorus, Irrigation) and crop production.
  - Visualization of correlations through scatter plots and a correlation heatmap.
- Machine Learning Model (Random Forest Regressor):
  - Splitting the data into training and testing sets.
  - Training a Random Forest Regressor model.
  - Evaluating the model's performance using Root Mean Squared Error (RMSE) and R-squared.
- Feature Importance Analysis:
  - Determining the importance of each feature using the Random Forest model's inherent feature importances.
  - Performing Permutation Importance to robustly assess feature relevance.
- Model Interpretation: Visualizing the architecture of one of the decision trees within the Random Forest.

## Key Findings
- Statistical Insights
  - The dataset contains 63 entries, with variables such as Precipitation, Deforestation, Carbon Dioxide, Methane, Temperature, Pesticides, Nitrogen, Phosphorus, Irrigation, and Production.
  - Outlier detection revealed a notable number of outliers in Irrigation (17), Phosphorus (11), and Pesticides (8), suggesting potential variability or unusual observations in these factors.
  - Correlation analysis showed strong positive correlations between Production and several factors, particularly Nitrogen, Phosphorus, Methane, and Carbon Dioxide.
- Machine Learning Results
  - The Random Forest Regressor achieved a Root Mean Squared Error (RMSE) of 6.34 and an R-squared score of 0.83 on the test set, indicating a reasonably good fit and predictive capability.
- Feature Importance
  - Both feature importance and permutation importance analyses consistently identified Nitrogen and Pesticides as the most critical drivers of crop production. Methane and Carbon Dioxide also showed significant importance.
  - Irrigation consistently appeared as the least important feature in both analyses, with a near-zero or slightly negative permutation importance, suggesting it might not be a strong predictor in this particular model setup.

## Dependencies
This project relies on the following Python libraries:
- pandas for data manipulation and analysis.
- numpy for numerical operations.
- matplotlib and seaborn for data visualization.
- sklearn (scikit-learn) for machine learning models (KNeighborsRegressor, RandomForestRegressor) and tools (train_test_split, mean_squared_error, r2_score, permutation_importance).
