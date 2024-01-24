import re
import numpy as np
from matplotlib import pyplot as plt
import os

from typing import Iterable

from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay


def file_to_tree_type_name(file_name: str, identifier: str) -> str:
    '''
    This function extracts the tree type name for a given geojson file.

    file_name: name of geojson file
    '''

    tree_type = re.search(f"([A-Z][a-z]+_[a-z]+)_{identifier}.geojson", file_name).group(1)
    return tree_type


def sample_file_to_tree_type(file_name: str) -> str:
    try:
        tree_type = re.search("([A-Z][a-z]+_[a-z]+)-", file_name).group(1)
    except:
        raise AttributeError(f"Error while parsing filename '{file_name}'")
    return tree_type


def determine_dimensions_from_collection(collection: dict):
    """
    Determines the spatial dimensions of an input geojson.
    """
    w, h, b = None, None, None
    for feature_ in collection["features"]:
        property_names = list(feature_["properties"].keys())
        for prop_name_ in property_names:
            rows = feature_["properties"][prop_name_]
            if rows is not None:
                b = len(feature_["properties"])
                h = len(feature_["properties"][prop_name_])        # number of rows
                w = len(feature_["properties"][prop_name_][0])     # number of columns (values per row)
                break
        if h is not None: break
    return w, h, b



def confusion_matrix_and_classf_metrics(y_true: Iterable, y_pred: Iterable, dataset, 
                                        outputForConfMatrix: str, verbose: bool = True, titleConfMatrix = 'Confusion Matrix'):
    '''
    This function calculates all the necessary metrics and saves the confusion matrix for a classification

    y_true: true label
    y_pred: predicted label
    dataset: dataset needs to have included the function label_to_labelname
    outputForConfMatrix: output folder for confusion matrix plot
    verbose: print metrics if True
    titleConfMatrix: title of confusion matrix + name of saved file
    '''

    # generate label names for confusion matrix later
    labels = []
    for label in np.sort(np.unique(y_true)):
        labels.append(dataset.label_to_labelname(label))

    acc = accuracy_score(y_true, y_pred)
    kapp = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division = np.nan)
    rec = recall_score(y_true, y_pred, average=None, zero_division = np.nan)
    #conf_mat = confusion_matrix(y_test, y_pred)

    prec_dict = dict(zip(labels, prec))
    rec_dict = dict(zip(labels, rec))

    if verbose: 
        print(
        f"\nMetrics:\n"       \
        f"\tAccuracy: {acc}\n"              \
        f"\tCohen kappa score: {kapp}\n"             \
        f"\tPrecision (Correctness): {prec_dict}\n"             \
        f"\tRecall (Completeness): {rec_dict}\n"             \
        )

    #fig, ax = plt.subplots(figsize=(15, 10))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels = labels, xticks_rotation = 'vertical', colorbar=True, cmap="Blues")
    disp.plot(colorbar=True, cmap="Blues")
    tick_size = 8
    plt.xticks(rotation=30, ha='right', size=tick_size)
    plt.yticks(size=tick_size)
    label_size = 18
    plt.xlabel("Predicted label", size=label_size)
    plt.ylabel("True label", size=label_size)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top = 0.99)#, left=0.4)
    plt.title(titleConfMatrix, size=label_size+2)
    fig = disp.ax_.get_figure() 
    fig.set_figwidth(10)
    fig.set_figheight(8)  

    if not os.path.isdir(outputForConfMatrix):
            os.makedirs(outputForConfMatrix, exist_ok=True)
    plt.savefig(os.path.join(outputForConfMatrix, titleConfMatrix + '.png'))