# -*- coding: utf-8 -*-
"""AUSTINE LOTANNA IHEJI._SportsPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15XbyXb_KOSZHSSczhS8nVdbSBADYH7MB

## **Importing libraries for use**
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle as pkl
import joblib as jb
from sklearn.impute import IterativeImputer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_predict

"""## **Reading from the CSV files**"""

male_legacy = pd.read_csv("male_players (legacy).csv", low_memory=False)

male_legacy.info()

"""## **1. Data Cleaning for both the Male legacy and FIFA 22 dataset**"""

for i in male_legacy.columns:
    if 'url' in i:
        male_legacy.drop(i,axis=1, inplace = True)

"""**Dropping columns with 30% or more null values**"""

# Dropping columns with 30% or more null values
male_legacy.dropna(thresh= 0.3 * len(players_21), axis=1, inplace=True)

male_legacy.head()

"""**Removing unncessary columns**"""

remove_columns = ['player_positions', 'player_id', 'fifa_version', 'fifa_update_date', 'short_name', 'long_name','club_team_id', 'club_name',
                  'league_name','age', 'dob', 'nationality_id', 'nationality_name' , 'real_face','club_position',
                  'club_jersey_number','ls', 'st', 'rs',
                   'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
                   'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb',
                   'rcb', 'rb', 'gk','player_traits', 'mentality_composure','release_clause_eur', 'club_joined_date' ]


male_legacy = male_legacy.drop(columns=remove_columns)

"""**Seperating the categorical and quantitative data**"""

alpha_df = male_legacy.select_dtypes(exclude='number')
numeric_df = male_legacy.select_dtypes(include='number')

"""## Working with the numeric data for Male Legacy"""

nan_count_columns = numeric_df.isna().sum()
nan_count_columns

"""**Imputing the numeric data**"""

from sklearn.impute import SimpleImputer

impute = SimpleImputer(strategy="median")
numeric_df = pd.DataFrame(impute.fit_transform(numeric_df), columns= numeric_df.columns)

#Taking out the iterative due to the low processing power of my machine
#imp = IterativeImputer(max_iter = 10, random_state = 0)
#numeric_df = pd.DataFrame(np.round(imp.fit_transform(numeric_df)), columns= numeric_df.columns)

nan_count_columns1 = numeric_df.isna().sum()
nan_count_columns1

"""## Working with the categorical data"""

nan_count_columns1 = alpha_df.isna().sum()
nan_count_columns1

"""**Encoding the categoricals**"""

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Create a copy of the DataFrame to avoid modifying the original data
encoded_df = alpha_df.copy()

for col in alpha_df.columns:
    encoded_df[col] = label_encoder.fit_transform(alpha_df[col])

alpha_df=encoded_df
alpha_df.head()

"""**Combining the dataframes**"""

relevant_data = pd.concat([alpha_df,numeric_df],axis=1)

relevant_data.info()

"""## **2. Correlation matrix to select the features and lables**"""

corr_matrix = relevant_data.corr()
corr_matrix['overall'].sort_values(ascending=False)

"""## **Confirming the correlation using graphs**"""

#showing a grapgh of overall against movement reactions
sns.scatterplot(x='movement_reactions', y='overall', data=relevant_data)

"""The plot highlights a direct and strong relationship between a player's movenemt reactions and their overall rating."""

#showing a grapgh of overall against potential
sns.scatterplot(x='potential', y='overall', data=relevant_data)

"""The plot highlights a direct and strong relationship between a player's potential and their overall rating, demonstrating that potential is a significant predictor of a player’s overall performance."""

#showing a grapgh of overall against skill_ball_control
sns.scatterplot(x='skill_ball_control', y='overall', data=relevant_data)

"""There is a visible positive trend; as skill_ball_control increases, the overall rating tends to increase as well.
This suggests that players with better ball control generally have higher overall ratings

**Selecting the feature subset that have the strongest correlation**
"""

main_21 = pd.DataFrame()
chosen_columns = ['overall', 'movement_reactions','potential','wage_eur','power_shot_power','value_eur','passing','mentality_vision',
                   'international_reputation', 'skill_long_passing', 'physic', 'skill_ball_control' ]

for i in chosen_columns:
    main_21[i] = relevant_data[i]
main_21

"""**Seperating the data into feature and labels**"""

y = main_21['overall']
X = main_21.drop('overall', axis = 1)

"""**Scaling the data**"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

jb.dump(scaler, 'scaler.pkl')

"""## **Method to clean the FIFA 22 data**"""

#creating the function for my players 22
def clean_players_22(df):
    for i in df.columns:
        if 'url' in i:
            df.drop(i,axis=1, inplace = True)

    df.dropna(thresh= 0.3 * len(df), axis=1, inplace=True)

    remove_columns = ['player_positions', 'sofifa_id', 'short_name', 'long_name','club_team_id', 'club_name',
    'league_name','age', 'dob','club_contract_valid_until', 'nationality_id', 'nationality_name' , 'real_face','club_position',
    'club_jersey_number','club_joined','ls', 'st', 'rs',
    'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm',
    'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb',
    'rcb', 'rb', 'gk','player_traits']

    df = df.drop(columns=remove_columns)

    alpha_df2 = df.select_dtypes(exclude='number')
    numeric_df2 = df.select_dtypes(include='number')

    numeric_df2 = pd.DataFrame(impute.fit_transform(numeric_df2), columns= numeric_df2.columns)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    # Create a copy of the DataFrame to avoid modifying the original data
    encoded_df2 = alpha_df2.copy()

    for col in alpha_df2.columns:
        encoded_df2[col] = label_encoder.fit_transform(alpha_df2[col])

    alpha_df2=encoded_df2

    relevant_data2 = pd.concat([alpha_df2,numeric_df2],axis=1)
    main_22 = pd.DataFrame()
    chosen_columns2 = ['overall', 'movement_reactions','potential','wage_eur','power_shot_power','value_eur','passing','mentality_vision',
    'international_reputation', 'skill_long_passing', 'physic', 'skill_ball_control']

    for i in chosen_columns2:
        main_22[i] = relevant_data2[i]

    y_test2 = main_22['overall']
    X_test2 = main_22.drop('overall', axis = 1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_test2 = scaler.fit_transform(X_test2)
    return X_test2, y_test2

"""## **3. Training the models**

**Splitting the data into training and testing**
"""

Xtrain , Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 0.2, random_state = 42)

"""**Initializing the models**"""

linear_model = LinearRegression()
xgb_model = XGBRegressor(n_estimators=600, learning_rate=0.1, max_depth=10)
dtree = DecisionTreeRegressor(max_depth=100, min_samples_split=40, min_samples_leaf=40)
random_forest_model = RandomForestRegressor(n_estimators=500, max_depth=40, min_samples_split=10, min_samples_leaf=40)
gradient_boosting_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=50)
AdaBoost = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)

def train_model(model, X, Y, cv=2):
    # Perform cross-validation
    y_pred = cross_val_predict(model, X, Y, cv=cv)

    # Fit the model on the whole dataset
    model.fit(X, Y)

    # Calculate and print evaluation metrics
    print(f"Model: {type(model).__name__}")
    print(f"Mean Absolute Error: {mean_absolute_error(Y, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(Y, y_pred)}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(Y, y_pred))}")
    print(f"R^2 Score: {r2_score(Y, y_pred)}")
    print()

"""**First training of the models with cross validation**"""

train_model(linear_model, Xtest, Ytest)

train_model(xgb_model, Xtest, Ytest)

train_model(dtree, Xtest, Ytest)

train_model(random_forest_model, Xtest, Ytest)

#train_model(gradient_boosting_model)

train_model(AdaBoost, Xtest, Ytest)

"""## **Creating an Ensemble model using the best four models**"""

# Creating the ensemble model with my best four models
ensemble_model = VotingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('dt', dtree),
        ('rf', random_forest_model),
        ('ada', AdaBoost)
    ]
)

# Train the ensemble model
ensemble_model.fit(Xtrain, Ytrain)

# Evaluate the ensemble on the test data
y_pred_ensemble = ensemble_model.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble, Ytest)}")
print()

"""## **4. Fine Tuning the components of the ensemble model**

**Performing a grid search with cross validation for XGBoost**
"""

# XGBoost parameters
param_grid_xgb = {
    'n_estimators': [100, 300, 600],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10],
}

# Create and fit the grid search
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=2, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_xgb.fit(Xtrain, Ytrain)

# Best parameters
best_params_xgb = grid_search_xgb.best_params_

print(f"Best parameters for XGBoost: {best_params_xgb}")

best_xgb = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_xgb, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_xgb, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_xgb, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_xgb, Ytest)}")
print()

"""**Performing a grid search with cross validation for decsion tree model**"""

# Decision Tree parameter
param_grid_dtree = {
    'max_depth': [200, 200],
    'min_samples_split': [20, 40],
    'min_samples_leaf': [20, 40]
}

# Create and fit the grid search
grid_search_dtree = GridSearchCV(estimator=dtree, param_grid=param_grid_dtree, cv=2, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_dtree.fit(Xtrain, Ytrain)

# Best parameters
best_params_dtree = grid_search_dtree.best_params_
print(f"Best parameters for Decision Tree: {best_params_dtree}")

best_dtree = grid_search_dtree.best_estimator_
y_pred_dtree = best_dtree.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_dtree, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_dtree, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_dtree, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_dtree, Ytest)}")
print()

"""**Performing a grid search with cross validation for the Random Forest Model**"""

# Random Forest parameters
param_grid_rf = {
    'n_estimators': [500, 400],
    'max_depth': [20, 30],
    'min_samples_split': [ 5, 10],
    'min_samples_leaf': [20, 40]
}

# Create and fit the grid search
grid_search_rf = GridSearchCV(estimator=random_forest_model,
                              param_grid=param_grid_rf,
                              cv= 2,
                              n_jobs=-1,
                              verbose=2,
                              scoring='neg_mean_squared_error')
grid_search_rf.fit(Xtrain, Ytrain)

# Best parameters
best_params_rf = grid_search_rf.best_params_
print(f"Best parameters for Random Forest: {best_params_rf}")

best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_rf, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_rf, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_rf, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_rf, Ytest)}")
print()

"""**Performing a grid search with cross validation for the ADA Boost regression Model**"""

# AdaBoost parameters
param_grid_ada = {
    'n_estimators': [500, 400],
    'learning_rate': [0.01, 0.1, 1.0],
}

# Create and fit the grid search
grid_search_ada = GridSearchCV(estimator=AdaBoost, param_grid=param_grid_ada, cv=2, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search_ada.fit(Xtrain, Ytrain)

# Best parameters
best_params_ada = grid_search_ada.best_params_
print(f"Best parameters for AdaBoost: {best_params_ada}")

best_ada = grid_search_ada.best_estimator_
y_pred_ada = best_ada.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ada, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ada, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ada, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ada, Ytest)}")
print()

"""**Combining all the best models from the grid search into an ensemble model using Voting Regressor**"""

# Creating the ensemble model
ensemble_model2 = VotingRegressor(
    estimators=[
        ('best_xgb', best_xgb),
        ('best_dtree', best_dtree),
        ('best_rf', best_rf),
        ('best_ada', best_ada )
    ]
)

# Train the ensemble model
ensemble_model2.fit(Xtrain, Ytrain)

# Evaluate the ensemble on the test data
y_pred_ensemble2 = ensemble_model2.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble2, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble2, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble2, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble2, Ytest)}")
print()

"""**Improving my previous random forest model and combining it with the voting regressor**"""

# Manually adjust hyperparameters for Random Forest and evaluating it
random_forest_model.set_params(n_estimators=400, max_depth=40, min_samples_split=2, min_samples_leaf=2)
random_forest_model.fit(Xtrain, Ytrain)


y_pred_ensemble3 = random_forest_model.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble3, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble3, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble3, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble3, Ytest)}")
print()

# Creating the ensemble model
ensemble_model4 = VotingRegressor(
    estimators=[
        ('best_xgb', best_xgb),
        ('best_dtree', best_dtree),
        ('best_rf', best_rf),
        ('rf', random_forest_model)
    ]
)

ensemble_model4.fit(Xtrain, Ytrain)

# Evaluate the ensemble on the test data
y_pred_ensemble4 = ensemble_model4.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble4, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble4, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble4, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble4, Ytest)}")
print()

"""**Improving my previous Gradient Booost model and combining it with the voting regressor**"""

# Manually adjust hyperparameters for Gradient Boosting and evaluate
gradient_boosting_model.set_params(n_estimators=400, learning_rate=0.02, max_depth=10)
gradient_boosting_model.fit(Xtrain, Ytrain)


y_pred_ensemble3 = gradient_boosting_model.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble3, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble3, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble3, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble3, Ytest)}")
print()

# Creating the ensemble model
ensemble_model5 = VotingRegressor(
    estimators=[
        ('best_xgb', best_xgb),
        ('best_dtree', best_dtree),
        ('best_rf', best_rf),
        ('rf', random_forest_model),
        ('gb', gradient_boosting_model)
    ]
)

ensemble_model5.fit(Xtrain, Ytrain)

# Evaluate the ensemble on the test data
y_pred_ensemble5 = ensemble_model5.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble5, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble5, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble5, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble5, Ytest)}")
print()

# Using the best xgboost as the final due to space restiction during deployment, its high accuracy and low MAE
ensemble_model6 = VotingRegressor(
    estimators=[
        ('best_xgb', best_xgb),
    ]
)

ensemble_model6.fit(Xtrain, Ytrain)

# Evaluate the ensemble on the test data
y_pred_ensemble6 = ensemble_model6.predict(Xtest)
print(f"Mean Absolute Error: {mean_absolute_error(y_pred_ensemble6, Ytest)}")
print(f"Mean Squared Error: {mean_squared_error(y_pred_ensemble6, Ytest)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred_ensemble6, Ytest))}")
print(f"R^2 Score: {r2_score(y_pred_ensemble6, Ytest)}")
print()

"""## **5. Testing on the new Data**"""

players_22 = pd.read_csv("players_22.csv",low_memory=False)
Xtest2 , Ytest2 = clean_players_22(players_22)

y_pred = ensemble_model6.predict(Xtest2)
# Calculate and print evaluation metrics
print(f"Model: {type(ensemble_model6).__name__}")
print(f"Mean Absolute Error: {mean_absolute_error(y_pred, Ytest2)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_pred, Ytest2))}")
print(f"R^2 Score: {r2_score(y_pred, Ytest2)}")
print()

jb.dump(ensemble_model6, 'main_final_model.pkl', compress = 9)

"""*******"""