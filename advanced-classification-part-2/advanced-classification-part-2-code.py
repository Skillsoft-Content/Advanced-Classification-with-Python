#######################################################
#######################################################
############    COPYRIGHT - DATA SOCIETY   ############
#######################################################
#######################################################

## ADVANCED CLASSIFICATION PART 2/ADVANCED CLASSIFICATION PART 2 ##

## NOTE: To run individual pieces of code, select the line of code and
##       press ctrl + enter for PCs or command + enter for Macs


#=================================================-
#### Slide 2: Loading packages  ####

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
#### Slide 6: Directory settings  ####

# Set 'main_dir' to location of the project folder
home_dir = Path(".").resolve()
main_dir = home_dir.parent.parent
print(main_dir)
data_dir = str(main_dir) + "/data"
print(data_dir)


#=================================================-
#### Slide 7: Load the cleaned dataset  ####

costa_clean = pickle.load(open(data_dir + "/costa_clean.sav","rb"))
print(costa_clean.head())


#=================================================-
#### Slide 8: Print info for our data  ####

costa_clean.columns


#=================================================-
#### Slide 9: Split into training and test sets  ####

# Select the predictors and target.
X = costa_clean.drop(['Target'], axis = 1)
y = np.array(costa_clean['Target'])

# Set the seed to 1.
np.random.seed(1)

# Split into the training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#=================================================-
#### Slide 10: Building our model  ####

forest = RandomForestClassifier(criterion = 'gini', 
                                n_estimators = 100, 
                                random_state = 1)


#=================================================-
#### Slide 11: Fitting our model  ####

# Fit the saved model to your training data.
forest.fit(X_train, y_train)


#=================================================-
#### Slide 12: Predicting with our data  ####

# Predict on test data.
y_predict_forest = forest.predict(X_test)

# Look at the first few predictions.
print(y_predict_forest[0:5,])


#=================================================-
#### Slide 15: Confusion matrix and accuracy  ####

# Take a look at test data confusion matrix.
conf_matrix_forest = metrics.confusion_matrix(y_test, y_predict_forest)
print(conf_matrix_forest)
accuracy_forest = metrics.accuracy_score(y_test, y_predict_forest)
print("Accuracy for random forest on test data: ", accuracy_forest)


#=================================================-
#### Slide 16: Accuracy of the training dataset  ####

# Compute accuracy using training data.
acc_train_forest = forest.score(X_train, y_train)

print ("Train Accuracy:", acc_train_forest)


#=================================================-
#### Slide 17: Evaluation of Random forest  ####

# Create a dictionary with accuracy values for our\ random forest model.
model_final_dict = {'metrics': ["accuracy"],
               'values':[round(accuracy_forest,4)],
                'model':['random_forest']}
model_final = pd.DataFrame(data = model_final_dict)
print(model_final)


#=================================================-
#### Slide 20: Subsetting our features  ####

costarica_features = costa_clean.drop('Target', axis = 1)
features = costarica_features.columns
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[0:10][::-1]

plt.figure(1)
plt.title('Feature Importance')
plt.barh(range(len(top_indices)), importances[top_indices], color = 'b', align = 'center')
labels = features[top_indices]
labels = [ '\n'.join(wrap(l,13)) for l in labels ]
plt.yticks(range(len(top_indices)), labels)
plt.xlabel('Relative Importance')


#=================================================-
#### Slide 23: Exercise 1  ####




#=================================================-
#### Slide 37: Boosting: build model  ####

# Save the parameters we will be using for our gradient boosting classifier.
gbm = GradientBoostingClassifier(n_estimators = 200, 
                                learning_rate = 1,
                                max_depth = 2, 
                                random_state = 1)


#=================================================-
#### Slide 38: Boosting: fit model  ####

# Fit the saved model to your training data.
gbm.fit(X_train, y_train)


#=================================================-
#### Slide 39: Boosting: predict  ####

# Predict on test data.
predicted_values_gbm = gbm.predict(X_test)
print(predicted_values_gbm)


#=================================================-
#### Slide 40: Confusion matrix and accuracy  ####

# Take a look at test data confusion matrix.
conf_matrix_boosting = metrics.confusion_matrix(y_test, predicted_values_gbm)
print(conf_matrix_boosting)
# Compute test model accuracy score.
accuracy_gbm = metrics.accuracy_score(y_test, predicted_values_gbm)
print('Accuracy of gbm on test data: ', accuracy_gbm)


#=================================================-
#### Slide 41: Accuracy of training model  ####

# Compute accuracy using training data.
train_accuracy_gbm = gbm.score(X_train, y_train)

print ("Train Accuracy:", train_accuracy_gbm)


#=================================================-
#### Slide 42: Pickle final accuracy  ####

# Add the model to our dataframe.
model_final_forest_gbm = model_final.append(
                        {'metrics' : "accuracy" ,
                        'values' : round(accuracy_gbm,4),
                        'model': 'boosting' } ,
                        ignore_index = True)

print(model_final_forest_gbm)


#=================================================-
#### Slide 43: Our top 10 features  ####

features = costarica_features.columns
importances = gbm.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[0:10][::-1]

plt.figure(1)
plt.title('Feature Importance')
plt.barh(range(len(top_indices)), importances[top_indices], color = 'b', align = 'center')
labels = features[top_indices]
labels = [ '\n'.join(wrap(l,13)) for l in labels ]
plt.yticks(range(len(top_indices)), features[top_indices])
plt.xlabel('Relative Importance')


#=================================================-
#### Slide 47: Exercise 2  ####




#######################################################
####  CONGRATULATIONS ON COMPLETING THIS MODULE!   ####
#######################################################
