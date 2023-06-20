#######################################################
#######################################################
############    COPYRIGHT - DATA SOCIETY   ############
#######################################################
#######################################################

## ADVANCED CLASSIFICATION PART 4/ADVANCED CLASSIFICATION PART 4 ##

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
#### Slide 8: Load the cleaned dataset and model metrics   ####

costa_clean = pickle.load(open(data_dir + "/costa_clean.sav","rb"))
metrics_gbm = pickle.load(open(data_dir + "/metrics_forest.sav","rb"))
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
#### Slide 11: Vanilla GBM model  ####

# Initialize GBM model.
gbm = GradientBoostingClassifier()

# Fit the model to train data.
gbm.fit(X_train, y_train)


#=================================================-
#### Slide 12: Convenience function for performance metrics  ####

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
#### Slide 13: Predict and evaluate vanilla GBM model  ####

# Predict on test data for GBM model.
gbm_y_predict = gbm.predict(X_test)

# Get prediction probabilities for the GBM model.
gbm_y_predict_proba = gbm.predict_proba(X_test)

# Get the GBM performance scores.
gbm_scores = get_performance_scores(y_test, gbm_y_predict, gbm_y_predict_proba)


#=================================================-
#### Slide 14: Precision vs recall curve: the tradeoff  ####

ax = plt.gca()
gbm_prec_recall = metrics.plot_precision_recall_curve(gbm, 
                                    X_test, 
                                    y_test,
                                    ax = ax,
                                    name = "GBM")

plt.show()


#=================================================-
#### Slide 15: ROC curve: the tradeoff  ####

ax = plt.gca()
gbm_roc = metrics.plot_roc_curve(gbm, 
                                 X_test, 
                                 y_test,
                                 ax = ax,
                                 name = "GBM")


plt.show()


#=================================================-
#### Slide 16: Append to other model performance scores   ####

metrics_gbm.update({"GBM": gbm_scores})
print(metrics_gbm)


#=================================================-
#### Slide 19: Randomized CV for GBM optimization: parameters  ####

gbm = GradientBoostingClassifier()
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
# Define learning rate parameters.
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]


#=================================================-
#### Slide 20: Randomized CV for GBM optimization: create grid  ####

# Create the random grid.
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}


#=================================================-
#### Slide 21: Randomized CV for GBM optimization: fit  ####

# Initialize the randomized search model just as you would any other scikit model.
gbm_random = RandomizedSearchCV(estimator = gbm, 
                                param_distributions = random_grid, 
                                n_iter = 100, 
                                cv = 3, 
                                verbose = 0, 
                                random_state = 1, 
                                n_jobs = -1)
                                
# Fit the random search model.
gbm_random.fit(X_train, y_train)
gbm_random = pickle.load(open(data_dir + "/gbm_random.sav","rb"))
gbm_random.best_params_


#=================================================-
#### Slide 22: Implement optimized GBM model  ####

# Pass parameters from randomized search to GBM classifier.
optimized_gbm = GradientBoostingClassifier(**gbm_random.best_params_)

# Fit model to train data.
optimized_gbm.fit(X_train, y_train)


#=================================================-
#### Slide 24: Exercise 1  ####




#=================================================-
#### Slide 26: Predict and evaluate optimized GBM model  ####

# Get class predictions.
optimized_gbm_y_predict = optimized_gbm.predict(X_test)

# Get prediction probabilities.
optimized_gbm_y_predict_proba = optimized_gbm.predict_proba(X_test)

# Compute performance metrics.
optimized_gbm_scores = get_performance_scores(y_test, 
                                              optimized_gbm_y_predict, 
                                              optimized_gbm_y_predict_proba)


#=================================================-
#### Slide 27: Precision vs recall curve: the tradeoff  ####

ax = plt.gca()
opt_gbm_prec_recall = metrics.plot_precision_recall_curve(optimized_gbm, 
                                    X_test, 
                                    y_test,
                                    ax = ax,
                                    name = "Optimized GBM")

gbm_prec_recall.plot(ax = ax, name = "GBM")
plt.show()


#=================================================-
#### Slide 28: ROC curve: the tradeoff  ####

ax = plt.gca()
opt_gbm_roc = metrics.plot_roc_curve(optimized_gbm, 
                                     X_test, 
                                     y_test,
                                     ax = ax,
                                     name = "Optimized GBM")


gbm_roc.plot(ax = ax, name = "GBM")
plt.show()


#=================================================-
#### Slide 29: Append to other model performance scores   ####

metrics_gbm.update({"Optimized GBM": optimized_gbm_scores})
print(metrics_gbm)



#=================================================-
#### Slide 30: Convert metrics dictionary to dataframe  ####

# Convert all metrics for each model to a dataframe.
metrics_gbm_df = pd.DataFrame(metrics_gbm)
metrics_gbm_df["metric"] = metrics_gbm_df.index
metrics_gbm_df = metrics_gbm_df.reset_index(drop = True)
print(metrics_gbm_df.head())


#=================================================-
#### Slide 31: Convert wide to long format  ####

metrics_gbm_long = pd.melt(metrics_gbm_df, 
                          id_vars = "metric",
                          var_name = "model",
                          value_vars = list(metrics_gbm.keys()))
print(metrics_gbm_long.head())


#=================================================-
#### Slide 32: Plot all models by metric  ####

# Create a 2X3 grid.
fig, axes = plt.subplots(2, 3, figsize = (12, 6))
# For each group in a grouped by metric dataframe, assign a metric to an axis object.
for (metric, group), ax in zip(metrics_gbm_long.groupby("metric"), axes.flatten()):
    # Plot each metric as a bar plot.
    group.plot(x = 'model',                                #<- model on x-axis
               y = 'value',                                #<- metric value on y-axis
               kind = 'bar',                               #<- bar plot 
               color = ["red", "green", "blue", "orange"], #<- color for each model
               ax = ax,                                    #<- axis object 
               title = metric,                             #<- plot title
               legend = None,                              #<- remove auto-legend
               sharex = True)                              #<- use the same x-axis
    ax.xaxis.set_tick_params(rotation = 45, labelsize=10)                #<- rotate labels for prettiness
plt.tight_layout(0.5)                                      #<- make sure no space is unused
plt.show()              


#=================================================-
#### Slide 34: Wrap comparison plot into a function  ####

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
        group.plot(x = 'model', y = 'value', kind = 'bar', color = colors, ax = ax,
                   title = metric,legend = None,sharex = True)
        ax.xaxis.set_tick_params(rotation = 45, labelsize=10)
    plt.tight_layout(0.5)
    return((fig, axes))


#=================================================-
#### Slide 35: Test function  ####

fig, axes = compare_metrics(metrics_gbm)
plt.show()


#=================================================-
#### Slide 37: Exercise 2  ####




#######################################################
####  CONGRATULATIONS ON COMPLETING THIS MODULE!   ####
#######################################################
