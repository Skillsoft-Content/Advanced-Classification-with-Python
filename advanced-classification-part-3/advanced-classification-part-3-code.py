#######################################################
#######################################################
############    COPYRIGHT - DATA SOCIETY   ############
#######################################################
#######################################################

## ADVANCED CLASSIFICATION PART 3/ADVANCED CLASSIFICATION PART 3 ##

## NOTE: To run individual pieces of code, select the line of code and
##       press ctrl + enter for PCs or command + enter for Macs


#=================================================-
#### Slide 2: Loading packages  ####

# Helper packages.
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt                     
import numpy as np
import math
from pathlib import Path

# Scikit-learn packages for building models and model evaluation.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics



#=================================================-
#### Slide 7: Directory settings  ####

# Set 'main_dir' to location of the project folder
home_dir = Path(".").resolve()
main_dir = home_dir.parent.parent
print(main_dir)
data_dir = str(main_dir) + "/data"
print(data_dir)


#=================================================-
#### Slide 8: Load the cleaned dataset  ####

costa_clean = pickle.load(open("costa_clean.sav","rb"))
print(costa_clean.head())


#=================================================-
#### Slide 9: Print info for our data  ####

costa_clean.columns


#=================================================-
#### Slide 10: Split into training and test sets  ####

# Select the predictors and target.
X = costa_clean.drop(['Target'], axis = 1)
y = np.array(costa_clean['Target'])

# Set the seed to 1.
np.random.seed(1)

# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#=================================================-
#### Slide 11: Recap: fit vanilla RF model  ####

# Initialize the classifier.
forest = RandomForestClassifier()

# Fit training data.
forest.fit(X_train, y_train)


#=================================================-
#### Slide 12: Recap: predict using vanilla RF model  ####

# Predict on test.
forest_y_predict = forest.predict(X_test)
print(forest_y_predict[:5])
#Predict on test, but instead of labels 
# we will get probabilities for class 0 and 1.
forest_y_predict_prob = forest.predict_proba(X_test) 
print(forest_y_predict_prob[5:])


#=================================================-
#### Slide 13: Recap: confusion matrix and accuracy  ####

# Compute accuracy for RF model.
forest_accuracy = metrics.accuracy_score(y_test, 
                                         forest_y_predict)
print(forest_accuracy)


#=================================================-
#### Slide 15: Precision  ####

forest_precision = metrics.precision_score(y_test, 
                                           forest_y_predict)
print(forest_precision)


#=================================================-
#### Slide 16: Recall  ####

forest_recall = metrics.recall_score(y_test, 
                                     forest_y_predict)
print(forest_recall)


#=================================================-
#### Slide 18: Precision-recall curve: the visual of tradeoff  ####

rf_prec_recall = metrics.plot_precision_recall_curve(forest, X_test, y_test, name = "RF")
plt.show()


#=================================================-
#### Slide 19: F1: precision vs recall   ####

forest_f1 = metrics.f1_score(y_test, 
                             forest_y_predict)
print(forest_f1)


#=================================================-
#### Slide 20: Fbeta: precision vs recall (weighted)  ####

forest_fbeta = metrics.fbeta_score(y_test, 
                                   forest_y_predict, 
                                   beta = 0.5)
print(forest_fbeta)


#=================================================-
#### Slide 24: Log loss using scikit-learn metrics  ####

# The second argument is an array of predicted probabilities, not labels!
forest_log_loss = metrics.log_loss(y_test, forest_y_predict_prob[:, 1], eps=1e-15)
print ("Log loss: ", forest_log_loss)


#=================================================-
#### Slide 25: Log loss using scikit-learn metrics - cont'd  ####

# Convert a difficult to interpret log loss to an overall accuracy in predicted probabilities.
print("Overall accuracy: ", math.exp(-forest_log_loss))


#=================================================-
#### Slide 26: Loss curve: ideal case  ####

# Probability values: 0 to 1 in 0.01 increments.
prob_increments = [x*0.01 for x in range(0, 101)]

# Evaluate predictions for when true value is 0.
loss_0 = [metrics.log_loss([0], [x], labels=[0,1]) for x in prob_increments]

# Evaluate predictions for when true value is 1.
loss_1 = [metrics.log_loss([1], [x], labels=[0,1]) for x in prob_increments]


#=================================================-
#### Slide 27: Loss curve: ideal case (cont'd)  ####

# Plot probability increments vs loss curves.
plt.plot(prob_increments, loss_0, label='true = 0')
plt.plot(prob_increments, loss_1, label='true = 1')
plt.legend()
plt.show()


#=================================================-
#### Slide 28: Loss curve: based on our data  ####

# Loss for predicting different fixed probability values.
losses = [metrics.log_loss(y_test, [y for x in range(len(y_test))]) for y in prob_increments]
# Plot predictions vs loss.
plt.plot(prob_increments, losses)
plt.show()
# Loss for predicting different fixed probability values.
losses = [metrics.log_loss(y_test, [y for x in range(len(y_test))]) for y in prob_increments]
# Plot predictions vs loss.
plt.plot(prob_increments, losses)
plt.show()


#=================================================-
#### Slide 31: ROC curve: the tradeoff  ####

rf_roc = metrics.plot_roc_curve(forest, X_test, y_test, name = "RF")
plt.show()


#=================================================-
#### Slide 32: AUC: area under the ROC curve  ####

# Where y_pred are probabilities and y_true are binary class labels
forest_auc = metrics.roc_auc_score(y_test, forest_y_predict_prob[:, 1])
print("AUC: ", forest_auc)


#=================================================-
#### Slide 33: Wrapping score evaluation into function  ####

def get_performance_scores(y_test, y_predict, y_predict_prob, eps=1e-15, beta=0.5):
    from sklearn import metrics
    # Scores keys.
    metric_keys = ["accuracy", "precision", "recall", "f1", "fbeta", "log_loss", "AUC"]
    # Score values.
    metric_values = [None]*len(metric_keys)
    metric_values[0] = metrics.accuracy_score(y_test, y_predict)
    metric_values[1] = metrics.precision_score(y_test, y_predict)
    metric_values[2] = metrics.recall_score(y_test, y_predict)
    metric_values[3] = metrics.f1_score(y_test, y_predict)
    metric_values[4] = metrics.fbeta_score(y_test, y_predict, beta=beta)
    metric_values[5] = metrics.log_loss(y_test, y_predict_prob[:, 1], eps=eps)
    metric_values[6] = metrics.roc_auc_score(y_test, y_predict_prob[:, 1])
    perf_metrics = dict(zip(metric_keys, metric_values))
    return(perf_metrics)


#=================================================-
#### Slide 34: Test score generating function  ####

forest_scores = get_performance_scores(y_test, forest_y_predict, forest_y_predict_prob)

metrics_forest = {"RF": forest_scores}
print(metrics_forest)


#=================================================-
#### Slide 36: Exercise 1  ####




#=================================================-
#### Slide 38: Vanilla RF model parameters  ####

forest.get_params()


#=================================================-
#### Slide 41: Parameter grid  ####

# Number of trees in random forest.
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 20)]

# Number of features to consider at every split.
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree.
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node.
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node.
min_samples_leaf = [1, 2, 4]

# Set Minimal Cost-Complexity Pruning parameter (has to be >= 0.0).
ccp_alpha = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3]

# Create the random grid 
# (a python dictionary in a form `'parameter_name': parameter_values`)
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'ccp_alpha': ccp_alpha}


#=================================================-
#### Slide 42: Set up RandomizedSearchCV function  ####

rf_random = RandomizedSearchCV(estimator = forest, #<- model object
                               param_distributions = random_grid, #<- param grid
                               n_iter = 100,#<- number of param. settings sampled 
                               cv = 3,      #<- 3-fold CV
                               verbose = 0, #<- silence lengthy output to console
                               random_state = 1, #<- set random state
                               n_jobs = -1)      #<- use all available processors
# Fit the random search model.
rf_random.fit(X_train, y_train) #<- fit like any other scikit-learn model
# Take a look at optimal combination of parameters.
print(rf_random.best_params_)


#=================================================-
#### Slide 44: Optimized RF model  ####

# Pass best parameters obtained through randomized search to RF classifier.
optimized_forest = RandomForestClassifier(**rf_random.best_params_)

# Train the optimized RF model.
optimized_forest.fit(X_train, y_train)


#=================================================-
#### Slide 45: Predict and compute performance scores  ####

# Get predicted labels for test data.
optimized_forest_y_predict = optimized_forest.predict(X_test)

# Get predicted probabilities.
optimized_forest_y_predict_proba = optimized_forest.predict_proba(X_test)
# Compute performance scores.
optimized_forest_scores = get_performance_scores(y_test, 
                                                 optimized_forest_y_predict,
                                                 optimized_forest_y_predict_proba)


#=================================================-
#### Slide 46: Precision vs recall curve: the tradeoff  ####

ax = plt.gca() #<- create a new axis object
opt_rf_prec_recall = metrics.plot_precision_recall_curve(optimized_forest,
                                 X_test, 
                                 y_test,
                                 ax = ax,
                                 name = "Optimized RF")
rf_prec_recall.plot(ax = ax, name = "RF") #<- add rf plot
plt.show()


#=================================================-
#### Slide 47: ROC curve: the tradeoff  ####

ax = plt.gca()
opt_rf_roc = metrics.plot_roc_curve(optimized_forest,
                       X_test,
                       y_test,
                       name = "Optimized RF",
                       ax = ax)

rf_roc.plot(ax = ax, name = "RF")
plt.show()


#=================================================-
#### Slide 48: Optimized RF: append to metrics dictionary  ####

metrics_forest.update({"Optimized RF": optimized_forest_scores})
print(metrics_forest)


#=================================================-
#### Slide 50: Exercise 2  ####




#######################################################
####  CONGRATULATIONS ON COMPLETING THIS MODULE!   ####
#######################################################
