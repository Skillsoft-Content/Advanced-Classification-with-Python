#######################################################
#######################################################
############    COPYRIGHT - DATA SOCIETY   ############
#######################################################
#######################################################

## ADVANCED CLASSIFICATION PART 1/ADVANCED CLASSIFICATION PART 1 ##

## NOTE: To run individual pieces of code, select the line of code and
##       press ctrl + enter for PCs or command + enter for Macs


#=================================================-
#### Slide 9: Loading packages  ####

import os
import pickle
import matplotlib.pyplot as plt                     
import numpy as np                                    
import pandas as pd
from pathlib import Path
from textwrap import wrap
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Random forest and boosting packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



#=================================================-
#### Slide 35: Directory settings  ####

# Set 'main_dir' to location of the project folder
home_dir = Path(".").resolve()
main_dir = home_dir.parent.parent
print(main_dir)
data_dir = str(main_dir) + "/data"
print(data_dir)


#=================================================-
#### Slide 36: Load the dataset  ####

household_poverty = pd.read_csv(data_dir + "/costa_rica_poverty.csv")
print(household_poverty.head())


#=================================================-
#### Slide 37: Converting the target variable  ####

household_poverty['Target'] = np.where(household_poverty['Target'] <= 3, 'vulnerable','non_vulnerable')
print(household_poverty['Target'].head())


#=================================================-
#### Slide 38: Data prep: target  ####

print(household_poverty.Target.dtypes)
household_poverty["Target"] = np.where(household_poverty["Target"] == "non_vulnerable", True, False)
# Check class again.
print(household_poverty.Target.dtypes)


#=================================================-
#### Slide 39: Data prep: check NAs  ####

household_poverty.isnull().sum()


#=================================================-
#### Slide 40: Subsetting the dataset  ####

costa_tree = household_poverty.drop(['household_id', 'ind_id', 'monthly_rent'], axis = 1)
print(costa_tree.head())


#=================================================-
#### Slide 42: Pickle the dataset  ####

pickle.dump(costa_tree, open(data_dir + "/costa_clean.sav", "wb" ))


#=================================================-
#### Slide 43: Exercise 1  ####




#=================================================-
#### Slide 46: Split into training and test sets  ####

# Select the predictors and target.
X = costa_tree.drop(['Target'], axis = 1)
y = np.array(costa_tree['Target'])

# Set the seed to 1.
np.random.seed(1)

# Split into the training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#=================================================-
#### Slide 48: Building our model  ####

forest = RandomForestClassifier(criterion = 'gini', 
                                n_estimators = 100, 
                                random_state = 1)


#=================================================-
#### Slide 49: Fitting our model  ####

# Fit the saved model to your training data.
forest.fit(X_train, y_train)


#=================================================-
#### Slide 50: Predicting with our data  ####

# Predict on test data.
y_predict_forest = forest.predict(X_test)

# Look at the first few predictions.
print(y_predict_forest[0:5,])


#=================================================-
#### Slide 53: Exercise 2  ####




#######################################################
####  CONGRATULATIONS ON COMPLETING THIS MODULE!   ####
#######################################################
