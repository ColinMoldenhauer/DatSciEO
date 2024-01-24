# A simple training script for sklearn models

import os
from re import M
import sys
import random
import time

from typing import Callable

import numpy as np

import utils

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

#import matplotlib.pyplot as plt

from joblib import dump, load


def train_sklearn_classifier(model: Callable, model_params: dict, dataset: utils.TreeClassifPreprocessedDataset, output_path: str, output_name: str, test_size: float = 0.3,
                            scaler: Callable | None = None, random_grid_params_optim: dict | None = None, n_random_grid_optim: float = 10, score_decision: str = 'accuracy',
                            verbose: bool = True, random_seed: int = 4):
    '''
    This function trains a given sklearn classifier from a given pytorch dataset, and saves the model as joblib-file. Random grid search is enabled.

    model: sklearn model
    model_params: model parameter in dictionary
    dataset: dataset in the form of a list of tuples consisting of two values: data and label
    output_path: relative path where model (and scaler) shall be saved after training
    output_name: name of model (and scaler) output
    test_size: size of test data [0,1]. Default: 0.3
    scaler: If None, dataset is not scaled. Default: None
    random_grid_params_optim: model parameter that shall be optimized in randmom grid search. If None, no hyperparamter optimization is performed
                              and all model parameter must be defined in <model_params>. Default: None
    n_random_grid_optim: number of parameter settings that are sampled. Trades off runtime vs quality of the solution. Only relevant if 
                         <random_grid_params_optim> is not None. Default: 10.
    score_decision: decision metric based on which the best model is selected from hyperparamter optimization. Only relevant if 
                    <random_grid_params_optim> is not None. Default: accuracy.
    verbose: If True, information is printed. Highly recommended to set it to True. Default: True.
    random_seed: random seed for reproducibility
    '''


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
        f"\tscaling: {scaler.__class__.__name__ if scaler is not None else False}\n"         \
        )

    if verbose: print(
        f"\nUsing sklearn model:\n"       \
        f"\tname: {model.__class__.__name__}\n"              \
        f"\tFixed hyperparameter: {model_params}\n"              \
        f"\thyperparameter optimization: {True if random_grid_params_optim else False}\n"             \
        )

    # set model parameter
    model_set = model.set_params(**model_params)

    # training
    print('\nTraining starts. This might take a while, especially if hyperparameter optimization is used ...')
    begin = time.time()

    if random_grid_params_optim: # using randomized search for hypterparater optimization
        rsv = RandomizedSearchCV(model_set, random_grid_params_optim, random_state=random_seed, n_iter=n_random_grid_optim, scoring=score_decision)
        rsv.fit(X_train, y_train)
        model_set = rsv.best_estimator_
        if verbose: print(
            f"\nBest model parameter from optimization:\n"       \
            f"\tParamter: {rsv.best_params_}\n"              \
            )
    else:
        model_set.fit(X_train, y_train) # using set parameters

    end = time.time()
    print('Training took ' + str(round(end-begin,2)) + ' s. hence, it took about ' + str(round((end-begin)/60,2)) + ' min.')

    # testing
    y_pred = model_set.predict(X_test)
    utils.confusion_matrix_and_classf_metrics(y_true=y_test, y_pred=y_pred, dataset=dataset, outputForConfMatrix=r'confusionMatrices', titleConfMatrix='Chris_Training')
    # acc = accuracy_score(y_test, y_pred)
    # kapp = cohen_kappa_score(y_test, y_pred)
    # prec = precision_score(y_test, y_pred, average=None, zero_division = np.nan)
    # rec = recall_score(y_test, y_pred, average=None, zero_division = np.nan)
    # #conf_mat = confusion_matrix(y_test, y_pred)

    # prec_dict = dict(zip(labels, prec))
    # rec_dict = dict(zip(labels, rec))

    # if verbose: 
    #     print(
    #     f"\nMetrics:\n"       \
    #     f"\tAccuracy: {acc}\n"              \
    #     f"\tCohen kappa score: {kapp}\n"             \
    #     f"\tPrecision (Correctness): {prec_dict}\n"             \
    #     f"\tRecall (Completeness): {rec_dict}\n"             \
    #     )


    # save model and scaler for later use. See https://stackoverflow.com/questions/53152627/saving-standardscaler-model-for-use-on-new-datasets/53153373#53153373
    dump(model_set, os.path.join(output_path, output_name + '_model.joblib')) 
    if verbose: print(f"\nSaving trained model to '{os.path.join(output_path, output_name + '_model.joblib')}'")
    if scaler is not None:
        dump(scaler, os.path.join(output_path, output_name + '_scaler.joblib')) 
        if verbose: print(f"Saving used scaler to '{os.path.join(output_path, output_name + '_scaler.joblib')}'")

    # disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels = labels, xticks_rotation = 'vertical')#, display_labels=dataset.classes)
    # plt.title('Confusion Matrix Training')
    # plt.tight_layout()
    # plt.show()




def test_sklearn_classifier(model_path: str, dataset: utils.TreeClassifPreprocessedDataset, scaler_path: str | None = None):
    '''
    This function trains a given sklearn classifier from a given pytorch dataset, and saves the model as joblib-file. Random grid search is enabled.

    model_path: path to saved sklearn model as joblib file
    dataset: dataset in the form of a list of tuples consisting of two values: data and label
    scaler_path: path to saved sklearn scaler as joblib file. If None, scaler is not applied. Default: None
    '''
    
    # load model
    loaded_model = load(model_path)

    # splitting into data and label
    X = np.array([x[0] for x in dataset])
    y = np.array([x[1] for x in dataset])

    # generate label names for confusion matrix later
    labels = []
    for label in np.sort(np.unique(y)):
        labels.append(dataset.label_to_labelname(label))


    # Sklearn needs a 2D vector (samples x features) as input
    X_original_shape = X.shape
    X = np.reshape(X, (X_original_shape[0], -1)) # samples x features

    # apply scaling
    if scaler_path is not None:
        loaded_scaler = load(scaler_path)
        X = loaded_scaler.fit_transform(X)

    # testing
    y_pred = loaded_model.predict(X)
    utils.confusion_matrix_and_classf_metrics(y_true=y, y_pred=y_pred, dataset=dataset, outputForConfMatrix=r'confusionMatrices', titleConfMatrix='RF Testing')
    # acc = accuracy_score(y, y_pred)
    # kapp = cohen_kappa_score(y, y_pred)
    # prec = precision_score(y, y_pred, average=None, zero_division = np.nan)
    # rec = recall_score(y, y_pred, average=None, zero_division = np.nan)
    # #conf_mat = confusion_matrix(y_test, y_pred)

    # prec_dict = dict(zip(labels, prec))
    # rec_dict = dict(zip(labels, rec))

    # if verbose: 
    #     print(
    #     f"\nMetrics:\n"       \
    #     f"\tAccuracy: {acc}\n"              \
    #     f"\tCohen kappa score: {kapp}\n"             \
    #     f"\tPrecision (Correctness): {prec_dict}\n"             \
    #     f"\tRecall (Completeness): {rec_dict}\n"             \
    #     )


    # disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels = labels, xticks_rotation = 'vertical')#, display_labels=dataset.classes)
    # plt.title('Confusion Matrix Testing')
    # plt.tight_layout()
    # plt.show()





if __name__ == '__main__':

    # Settings for training --------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------

    random_seed = 4 # random seed for reproducibility
    verbose = True
    output_path = 'models' # path where model parameters should be saved
    output_name = '0122chris' # name of file that stores model
    # See https://scikit-learn.org/stable/model_persistence.html
    # See https://stackoverflow.com/questions/53152627/saving-standardscaler-model-for-use-on-new-datasets/53153373#53153373


    # Dataset parameters -----------------------------------------------
    dataset_dir = r"data/dataColin/train_val_delete_nan_samples" # location of data
    # create dataset
    dataset = utils.TreeClassifPreprocessedDataset(
        dataset_dir,
        torchify=False,)
        #indices=[*random.sample(range(1, 208734), 20000)])
    test_size = 0.3 # test size
    scaler = StandardScaler() # scaler for input data. If none is required, put scaler = None


    # Sklearn model parameters -----------------------------------------
    model = RandomForestClassifier() # sklearn model
    params = {'n_jobs' : -1,
            'random_state': random_seed,}
    '''
    'bootstrap': False,
    'max_depth': 20,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 100} # fixed hyperparam.
    '''


    # Random Grid Search Parameters ------------------------------------
    # if no random grid search is desired, random_grid should be an empty dicctionary
    n_optimizations = 20 # number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution.
    random_grid = {'bootstrap': [True, False],
                'max_depth': [10, 20, 50, 100, None],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]} # hyperparameter to be optimized by random grid search
    #random_grid = {}
    score = 'accuracy' # score to define best model according to hyperparameter optimization; 
    # see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


    # Training ---------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    training = False

    if training:
        train_sklearn_classifier(model = model, model_params = params, dataset = dataset, output_path = output_path, output_name = output_name, test_size = test_size,
                                scaler = scaler, random_grid_params_optim = random_grid, n_random_grid_optim = n_optimizations, score_decision = score,
                                verbose = verbose, random_seed = random_seed)



    # Settings for testing ---------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    
    model_path = r'models/0122chris_model.joblib'
    scaler_path = r'models/0122chris_scaler.joblib'

    dataset_dir = r"data/dataColin/test_delete_nan_samples" # location of data
    #dataset_dir = r"data/dataColin/train_val_delete_nan_samples"
    # create dataset
    dataset = utils.TreeClassifPreprocessedDataset(dataset_dir, torchify=False)#, indices=[*random.sample(range(1, 208734), 20000)])
    

    # Testing ----------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------
    testing = True

    if testing:
        test_sklearn_classifier(model_path=model_path, scaler_path=scaler_path, dataset=dataset)