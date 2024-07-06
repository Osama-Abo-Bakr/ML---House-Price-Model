# House Price Prediction

## Project Overview

This project aims to predict house prices using various machine learning models. The focus is on cleaning the dataset, engineering features, and building and tuning models to achieve high accuracy.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)
9. [Contact](#contact)

## Introduction

Accurately predicting house prices is crucial for stakeholders in the real estate market. This project uses machine learning algorithms to model house prices based on various features from a dataset.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning modeling and evaluation
- **XGBoost**: Extreme Gradient Boosting

## Data Preprocessing

1. **Data Loading**:
   - Loaded the training and test datasets using `pd.read_csv()`.

2. **Cleaning Data**:
   - Removed irrelevant columns using the `clean_data()` function.
   - Handled missing values by filling numerical columns with the median and categorical columns with the mode using the `fillna_data()` function.

3. **Label Encoding**:
   - Applied `LabelEncoder` to convert categorical features to numerical values.

4. **Exploratory Data Analysis**:
   - Used `sns.heatmap()` and `sns.relplot()` for visualizing correlations and relationships.

## Modeling

1. **Random Forest Regressor**:
   - Configured with `n_estimators=100` and `max_depth=3000`.
   - Trained on the entire dataset and evaluated using `.score()`.

2. **XGBoost Regressor**:
   - Set with `n_estimators=10000`, `max_depth=100`, and `learning_rate=0.01`.
   - Trained and evaluated for predictive accuracy.

3. **AdaBoost Regressor**:
   - Implemented with a `DecisionTreeRegressor` and `RandomForestRegressor` as base estimators.
   - Configured with different parameters for depth and learning rate, and trained on the dataset.

## Results

- **Random Forest Regressor**:
  - Training Score: 0.9825128488400557

- **XGBoost Regressor**:
  - Training Score: 0.9999999999061427

- **AdaBoost Regressor with Decision Tree**:
  - Training Score: 0.9999871324164955

- **AdaBoost Regressor with Random Forest**:
  - Training Score: 0.9917741176822643

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/house-price-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebook or script to view the results.

## Usage

1. **Prepare Data**:
   - Ensure the dataset is available at the specified path.

2. **Train Models**:
   - Run the provided script or Jupyter notebook to train models and evaluate performance.

3. **Predict Outcomes**:
   - Use the trained model to predict house prices by providing new input data.

## Conclusion

This project successfully demonstrates the application of machine learning in predicting house prices. The models developed achieve high accuracy and can be used for practical real estate analytics.

--

### Sample Code (for reference)

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ID = test["Id"]

# Data cleaning function
def clean_data(data):
    data = data.drop(["MiscFeature", "Fence", "PoolQC", "Alley", "Id", "FireplaceQu", "MasVnrType"], axis=1)
    return data

# Apply cleaning
train = clean_data(train)
test = clean_data(test)

# Fill missing values
def fillna_data(data):
    data_not_object = data.select_dtypes(exclude=["object"])
    data_object = data.select_dtypes(include=["object"])
    for col in data_not_object:
        data[col].fillna(data[col].median(), inplace=True)
    for col in data_object:
        data[col].fillna(data[col].value_counts().index[0], inplace=True)
    return data

train = fillna_data(train)
test = fillna_data(test)

# Visualize correlations
plt.figure(figsize=(50, 20))
sns.heatmap(train.corr(), annot=True)
plt.show()

# Label Encoding
object_element = train.select_dtypes(include=["object"])
la = LabelEncoder()
for col in object_element.columns:
    train[col] = la.fit_transform(train[col])
    test[col] = la.transform(test[col])

# Prepare input and output
X_input = train.drop(["SalePrice"], axis=1)
Y_output = train["SalePrice"]

# Model 1: Random Forest Regressor
model_Random = RandomForestRegressor(n_estimators=100, max_depth=3000)
model_Random.fit(X_input, Y_output)
print(f"Random Forest Regressor - Train Score: {model_Random.score(X_input, Y_output)}")

# Model 2: XGBoost Regressor
model_xgb = xgb.XGBRegressor(n_estimators=10000, max_depth=100, learning_rate=0.01)
model_xgb.fit(X_input, Y_output)
print(f"XGBoost Regressor - Train Score: {model_xgb.score(X_input, Y_output)}")

# Model 3: AdaBoost Regressor with Decision Tree
Adaboost_reg = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3000), n_estimators=500, learning_rate=0.1)
Adaboost_reg.fit(X_input, Y_output)
print(f"AdaBoost Regressor with Decision Tree - Train Score: {Adaboost_reg.score(X_input, Y_output)}")

# Model 4: AdaBoost Regressor with Random Forest
Adaboost_reg1 = AdaBoostRegressor(estimator=RandomForestRegressor(max_depth=3000), n_estimators=10, learning_rate=0.5)
Adaboost_reg1.fit(X_input, Y_output)
print(f"AdaBoost Regressor with Random Forest - Train Score: {Adaboost_reg1.score(X_input, Y_output)}")
```

Feel free to explore and contribute to the project. Your insights are valuable!

---
