import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error

#1. Loading Data set
housing = pd.read_csv("housing.csv")


#2. Create Stratified test set
housing['income_cat'] = pd.cut(housing["median_income"],
                bins = (0, 1.5, 3.0, 4.5, 6.0, np.inf),
                labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis =1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis =1)


#Working on training data copy
housing = strat_train_set.copy()


#3. Seprate Features and Labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis = 1)


#4. Seprate numerical and categorical data
num_attributes = housing.drop("ocean_proximity", axis = 1).columns.to_list()
cat_attributes = ["ocean_proximity"]

#5. Making pipeline for
# Numerical columns
num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy= "median")),
                ("scaler", StandardScaler())
                ])

# Categorical coluns
cat_pipeline = Pipeline([
                ("encoder", OneHotEncoder())
                ])

#Full pipline
full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attributes),
                ('cat', cat_pipeline, cat_attributes)
                ])

#6. Transform Data into ful pipelije
data_prepared = full_pipeline.fit_transform(housing)
# print(data_prepared)

#7. Train the model
#Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, housing_labels)
lin_preds = lin_reg.predict(data_prepared)
lin_rsmes = -cross_val_score(lin_reg, data_prepared, housing_labels,scoring="neg_root_mean_squared_error", cv = 10)
print(f"Cross validation description of Linear regression model:\n{pd.Series(lin_rsmes).describe()}")
# lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
# print(f"The root mean squared error for Linear Regression is {lin_rmse}")

#Decision Tree Regressor
dec_tree = DecisionTreeRegressor()
dec_tree.fit(data_prepared, housing_labels)
dec_preds = dec_tree.predict(data_prepared)
dec_rsmes = -cross_val_score(dec_tree, data_prepared, housing_labels,scoring="neg_root_mean_squared_error", cv = 10)
print(f"Cross validation description of Decision Tree Regressor:\n{pd.Series(dec_rsmes).describe()}")
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds)
# print(f"The root mean squared error for Decision Tree Regressor is {dec_rmse}")

#Random Forest Regressor
ran_tree = RandomForestRegressor()
ran_tree.fit(data_prepared, housing_labels)
ran_preds = ran_tree.predict(data_prepared)
ran_rsmes = -cross_val_score(ran_tree, data_prepared, housing_labels,scoring="neg_root_mean_squared_error", cv = 10)
print(f"Cross validation description of Random Forest Regressor:\n{pd.Series(ran_rsmes).describe()}")
# ran_rmse = root_mean_squared_error(housing_labels, ran_preds)
# print(f"The root mean squared error for Random Forest Classifier is {ran_rmse}") 