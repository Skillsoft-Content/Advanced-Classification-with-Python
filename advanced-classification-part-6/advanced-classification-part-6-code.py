#######################################################
#######################################################
############    COPYRIGHT - DATA SOCIETY   ############
#######################################################
#######################################################

## ADVANCED CLASSIFICATION PART 6/ADVANCED CLASSIFICATION PART 6 ##

## NOTE: To run individual pieces of code, select the line of code and
##       press ctrl + enter for PCs or command + enter for Macs


#=================================================-
#### Slide 2: Loading packages  ####

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
#### Slide 12: Directory settings  ####

# Set 'main_dir' to location of the project folder
home_dir = Path(".").resolve()
main_dir = home_dir.parent.parent
print(main_dir)
data_dir = str(main_dir) + "/data"
print(data_dir)


#=================================================-
#### Slide 13: Load the cleaned dataset  ####

costa_clean = pickle.load(open((data_dir + "/costa_clean.sav"),"rb"))
# Print the head.
print(costa_clean.head())


#=================================================-
#### Slide 14: Print info on data  ####

# Print the columns.
costa_clean.columns


#=================================================-
#### Slide 15: Split into training and test sets  ####

# Select the predictors and target.
X = costa_clean.drop(['Target'], axis = 1)
y = np.array(costa_clean['Target'])

# Set the seed to 1.
np.random.seed(1)

# Split into training and test sets with 70% train and 30% test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


#=================================================-
#### Slide 16: Build a SVM model  ####

# Build the SVM model.
# Note here that the kernel rbf means radial kernel.
sv_machine = SVC(kernel = 'rbf', 
                  gamma = 0.011,
                  probability = True)
sv_machine.fit(X_train, y_train)


#=================================================-
#### Slide 17: Predict on the test dataset  ####

# Predict on the test dataset.
svm_y_predict = sv_machine.predict(X_test)
svm_y_predict[0:5]

# Predict on test, but instead of labels 
# we will get probabilities for class 0 and 1.
svm_y_predict_prob = sv_machine.predict_proba(X_test) 
print(svm_y_predict_prob[5:])


#=================================================-
#### Slide 18: Recap: score evaluation function  ####

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
#### Slide 19: Predict on the test dataset  ####

svm_scores = get_performance_scores(y_test, svm_y_predict, svm_y_predict_prob)
print(svm_scores)


#=================================================-
#### Slide 20: Precision vs recall curve: the tradeoff  ####

ax = plt.gca() #<- create a new axis object
svm_prec_recall = metrics.plot_precision_recall_curve(sv_machine,
                                 X_test, 
                                 y_test,
                                 ax = ax,
                                 name = "SVM")

plt.show()
ax = plt.gca() #<- create a new axis object
svm_prec_recall = metrics.plot_precision_recall_curve(sv_machine,
                                 X_test, 
                                 y_test,
                                 ax = ax,
                                 name = "SVM")
plt.show()


#=================================================-
#### Slide 21: ROC curve: the tradeoff  ####

ax = plt.gca()
svm_roc = metrics.plot_roc_curve(sv_machine,
                       X_test,
                       y_test,
                       name = "SVM",
                       ax = ax)

plt.show()
ax = plt.gca()
svm_roc = metrics.plot_roc_curve(sv_machine,
                       X_test,
                       y_test,
                       name = "SVM",
                       ax = ax)


plt.show()


#=================================================-
#### Slide 22: Save final metrics of SVM  ####

metrics_svm = pickle.load(open((data_dir + "/metrics_svm.sav"),"rb")) 


#=================================================-
#### Slide 23: Save final metrics of SVM  ####

# Add the model to our dataframe.
metrics_svm.update({"SVM": svm_scores})
print(metrics_svm)


#=================================================-
#### Slide 25: Exercise 1  ####




#=================================================-
#### Slide 28: Grid search and cv on SVM model  ####

# Set the parameters by cross-validation.
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
                     
# Fit the tuned parameters for grid search with 5 fold cross-validation to SVM. 
svm_cv = GridSearchCV(SVC(), tuned_parameters, cv = 5)
svm_cv.fit(X_train, y_train)


#=================================================-
#### Slide 29: Find best parameters  ####

# Find the best tuned parameters.
print(svm_cv.best_params_)
# Extract the best hyperparameters.
optimized_c = svm_cv.best_params_['C']
optimized_gamma = svm_cv.best_params_['gamma']
optimized_kernel = svm_cv.best_params_['kernel']


#=================================================-
#### Slide 30: Fit the best parameters to build the optimized model  ####

# Run the model with optimized hyperparameters.
sv_cv_optimized = SVC(kernel = optimized_kernel,
                      gamma = optimized_gamma, 
                      C = optimized_c,
                      probability = True)

sv_cv_optimized.fit(X_train, y_train)


#=================================================-
#### Slide 31: Predict on the test dataset  ####

# Predict on the test dataset.
opt_svm_y_predict = sv_cv_optimized.predict(X_test)
opt_svm_y_predict[0:5]

# Predict on test, but instead of labels 
# we will get probabilities for class 0 and 1.
opt_svm_y_predict_prob = sv_cv_optimized.predict_proba(X_test) 
print(opt_svm_y_predict_prob[5:])


#=================================================-
#### Slide 32: Predict on the test dataset  ####

opt_svm_scores = get_performance_scores(y_test, opt_svm_y_predict, opt_svm_y_predict_prob)
print(opt_svm_scores)


#=================================================-
#### Slide 33: Precision vs recall curve: the tradeoff  ####

ax = plt.gca() #<- create a new axis object
opt_svm_prec_recall = metrics.plot_precision_recall_curve(sv_cv_optimized,
                                 X_test, 
                                 y_test,
                                 ax = ax,
                                 name = "Optimized SVM")

svm_prec_recall.plot(ax = ax, name = "SVM") #<- add rf plot
plt.show()
ax = plt.gca() #<- create a new axis object
opt_svm_prec_recall = metrics.plot_precision_recall_curve(sv_cv_optimized,
                                 X_test, 
                                 y_test,
                                 ax = ax,
                                 name = "Optimized SVM")

_ = svm_prec_recall.plot(ax = ax, name = "SVM") #<- add rf plot
plt.show()


#=================================================-
#### Slide 34: ROC curve: the tradeoff  ####

ax = plt.gca()
opt_svm_roc = metrics.plot_roc_curve(sv_cv_optimized,
                       X_test,
                       y_test,
                       name = "Optimized SVM",
                       ax = ax)

svm_roc.plot(ax = ax, name = "SVM")
plt.show()
ax = plt.gca()
opt_svm_roc = metrics.plot_roc_curve(sv_cv_optimized,
                       X_test,
                       y_test,
                       name = "Optimized SVM",
                       ax = ax)

_ = svm_roc.plot(ax = ax, name = "SVM")
plt.show()


#=================================================-
#### Slide 35: Save final metrics of SVM  ####

# Add the model to our dataframe.
metrics_svm.update({"Optimized SVM": opt_svm_scores})
print(metrics_svm)


#=================================================-
#### Slide 36: Create comparison plot using function  ####

def compare_metrics(metrics_dict, color_list = None):
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df["metric"] = metrics_df.index
    metrics_df = metrics_df.reset_index(drop = True)
    metrics_long = pd.melt(metrics_df, id_vars = "metric", var_name = "model",
                           value_vars = list(metrics_dict.keys()))
    if color_list is None:
        cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = cmap[:len(metrics_dict.keys())]
    else:
        colors = color_list
    fig, axes = plt.subplots(2, 3, figsize = (12, 6))
    for (metric, group), ax in zip(metrics_long.groupby("metric"), axes.flatten()):
        group.plot(x = 'model', y = 'value', kind = 'bar',color = colors,ax = ax,
                   title = metric,legend = None,sharex = True)
        ax.xaxis.set_tick_params(rotation = 45,labelsize=7)
    plt.tight_layout(0.5)
    return((fig, axes))


#=================================================-
#### Slide 37: Create comparison plot using function  ####

fig, axes = compare_metrics(metrics_svm)
plt.show()
fig, axes = compare_metrics(metrics_svm)
plt.show()


#=================================================-
#### Slide 41: Exercise 2  ####




#######################################################
####  CONGRATULATIONS ON COMPLETING THIS MODULE!   ####
#######################################################
