# A simple training script for sklearn models

import os
import sys
import random

import numpy as np

from utils import TreeClassifPreprocessedDataset

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from joblib import dump, load


###########################################################
##################      SETTINGS        ###################
###########################################################


random_seed = 4 # random seed for reproducibility
verbose = True
safe_models = True # if True, models and scalers are safed in models folder
# See https://scikit-learn.org/stable/model_persistence.html
# See https://stackoverflow.com/questions/53152627/saving-standardscaler-model-for-use-on-new-datasets/53153373#53153373
path_save = 'models' # path where model parameters should be saved
name_saved_model = 'chris_model' # name of file that stores model
name_saved_scaler = 'chris_scaler' # name of file that stores used scaler for prediction

# Dataset parameters -----------------------------------------------
dataset_dir = r"data/1102_delete_nan_samples_B2" # location of data
# create dataset
dataset = TreeClassifPreprocessedDataset(
    dataset_dir,
    torchify=False,
    indices=[*random.sample(range(1, 14000), 2000)])
test_size = 0.3 # test size
scaler = StandardScaler() # scaler for input data. If none is required, put scaler = None

# Sklearn model parameters -----------------------------------------
model = RandomForestClassifier() # sklearn model
params = {'n_jobs' : -1,
          'random_state': random_seed,
          'bootstrap': False,
          'max_depth': 20,
          'min_samples_leaf': 2,
          'min_samples_split': 2,
          'n_estimators': 100} # fixed hyperparam.

# Random Grid Search Parameters ------------------------------------
# if no random grid search is desired, random_grid should be an empty dicctionary
n_optimizations = 10 # number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
random_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 50, 100, None],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500]} # hyperparameter to be optimized by random grid search
random_grid = {}
scores = 'accuracy' # score to define best model according to hyperparameter optimization; 
# see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


###########################################################
##############      SETTINGS END        ###################
###########################################################




# splitting into data and label
X = np.array([x[0] for x in dataset])
y = np.array([x[1] for x in dataset])


# Sklearn needs a 2D vector (samples x features) as input
X_original_shape = X.shape
X = np.reshape(X, (X_original_shape[0], -1)) # samples x features

# apply scaling
if scaler is not None:
    X = scaler.fit_transform(X)

# splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=True)

# generate label names for confusion matrix later
labels = []
for label in np.sort(np.unique(y_test)):
    labels.append(dataset.label_to_labelname(label))


if verbose: print(
    f"\nUsing dataset with properties:\n"       \
    f"\tsamples: {len(dataset)}\n"              \
    f"\t\ttrain: {len(X_train)}\n"             \
    f"\t\ttest:  {len(X_test)}\n"               \
    f"\tshape: {dataset[0][0].shape}\n"         \
    f"\tscaling: {True if scaler is not None else False}\n"         \
    )

if verbose: print(
    f"\nUsing sklearn model:\n"       \
    f"\tname: {model.__class__.__name__}\n"              \
    f"\thyperparameter optimization: {True if random_grid else False}\n"             \
    )

# set model parameter
model_set = model.set_params(**params)

# training
if random_grid: # using randomized search for hypterparater optimization
    rsv = RandomizedSearchCV(model_set, random_grid, random_state=random_seed, n_iter=n_optimizations)
    rsv.fit(X_train, y_train)
    model_set = rsv.best_estimator_
else:
    model_set.fit(X_train, y_train) # using set parameters


# testing
y_pred = model_set.predict(X_test)
acc = accuracy_score(y_test, y_pred)
kapp = cohen_kappa_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
#conf_mat = confusion_matrix(y_test, y_pred)

if verbose: 
    print(
    f"\nMetrics:\n"       \
    f"\tAccuracy: {acc}\n"              \
    f"\tCohen kappa score: {kapp}\n"             \
    f"\tPrecision (Correctness): {prec}\n"             \
    f"\tRecall (Completeness): {rec}\n"             \
    )
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels = labels)#, display_labels=dataset.classes)
    disp.plot()
    plt.show()


if safe_models: # save scaler for later use. See https://stackoverflow.com/questions/53152627/saving-standardscaler-model-for-use-on-new-datasets/53153373#53153373
        dump(model_set, os.path.join(path_save, name_saved_model + '.joblib')) 
        if verbose: print(f"\nSaving trained model to '{os.path.join(path_save, name_saved_model + '.joblib')}'")
        if scaler is not None:
            dump(scaler, os.path.join(path_save, name_saved_scaler + '.joblib')) 
            if verbose: print(f"Saving used scaler to '{os.path.join(path_save, name_saved_scaler + '.joblib')}'")
