#######################################################
#######################################################
############    COPYRIGHT - DATA SOCIETY   ############
#######################################################
#######################################################

## ADVANCED CLASSIFICATION PART 5/ADVANCED CLASSIFICATION PART 5 ##

## NOTE: To run individual pieces of code, select the line of code and
##       press ctrl + enter for PCs or command + enter for Macs


#=================================================-
#### Slide 14: Loading packages  ####

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


#=================================================-
#### Slide 22: Directory settings  ####

# Set 'main_dir' to location of the project folder
home_dir = Path(".").resolve()
main_dir = home_dir.parent.parent
print(main_dir)
data_dir = str(main_dir) + "/data"
print(data_dir)


#=================================================-
#### Slide 23: Load the cleaned dataset  ####

costa_clean = pickle.load(open((data_dir + "/costa_clean.sav"),"rb"))
# Print the head.
print(costa_clean.head())


#=================================================-
#### Slide 24: Print info on data  ####

# Print the columns.
costa_clean.columns


#=================================================-
#### Slide 25: Split into training and test sets  ####

# Select the predictors and target.
X = costa_clean.drop(['Target'], axis = 1)
y = np.array(costa_clean['Target'])

# Set the seed to 1.
np.random.seed(1)

# Split into training and test sets with 70% train and 30% test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#=================================================-
#### Slide 28: Support vector classifier model  ####

# Create an SVC classifier.
svclassifier = SVC(kernel = 'linear', 
                   probability = True)

# Fit the model.
svclassifier.fit(X_train, y_train)


#=================================================-
#### Slide 29: Recap: score evaluation function  ####

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
#### Slide 30: Predict on the test dataset  ####

# Predict on the test dataset.
svc_y_predict = svclassifier.predict(X_test)
svc_y_predict[0:5]

#Predict on test, but instead of labels 
# we will get probabilities for class 0 and 1.
svc_y_predict_prob = svclassifier.predict_proba(X_test) 
print(svc_y_predict_prob[5:])


#=================================================-
#### Slide 31: Predict on the test dataset  ####

svc_scores = get_performance_scores(y_test, svc_y_predict, svc_y_predict_prob)
print(svc_scores)


#=================================================-
#### Slide 32: Precision vs recall curve: the tradeoff  ####

svc_prec_recall = metrics.plot_precision_recall_curve(svclassifier, 
                               X_test, y_test, 
                               name = "SVC")
plt.show()
svc_prec_recall = metrics.plot_precision_recall_curve(svclassifier, X_test, y_test, name = "SVC")
plt.show()


#=================================================-
#### Slide 33: ROC curve: the tradeoff  ####

svc_roc = metrics.plot_roc_curve(svclassifier,
                                X_test,
                                y_test,
                                name = "SVC")
plt.show()
svc_roc = metrics.plot_roc_curve(svclassifier,
                                X_test,
                                y_test,
                                name = "SVC")
plt.show()


#=================================================-
#### Slide 34: Save final metrics of SVC  ####

metrics_svm = pickle.load(open((data_dir + "/metrics_gbm.sav"),"rb"))


#=================================================-
#### Slide 35: Save final metrics of SVC  ####

# Add the model to our dataframe.
metrics_svm.update({"SVC": svc_scores})
print(metrics_svm)


#=================================================-
#### Slide 37: Exercise 1  ####




#######################################################
####  CONGRATULATIONS ON COMPLETING THIS MODULE!   ####
#######################################################
