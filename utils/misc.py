import json
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
                                        output_folder: str, verbose: bool = True,
                                        titleConfMatrix='Confusion Matrix',
                                        filename=None):
    '''
    This function calculates all the necessary metrics and saves the confusion matrix for a classification

    y_true: true label
    y_pred: predicted label
    dataset: dataset needs to have included the function label_to_labelname
    output_folder: output folder for confusion matrix plot
    verbose: print metrics if True
    titleConfMatrix: title of confusion matrix + (optionally) name of saved file
    filename: File where to save confusion matrix image
    '''
    os.makedirs(output_folder, exist_ok=True)

    labels = dataset.classes

    acc = accuracy_score(y_true, y_pred)
    kapp = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None, zero_division = np.nan)
    rec = recall_score(y_true, y_pred, average=None, zero_division = np.nan)

    prec_dict = dict(zip(labels, prec))
    rec_dict = dict(zip(labels, rec))

    metrics = {
        "accuracy": acc,
        "kappa": kapp,
        "precision": prec_dict,
        "recall": rec_dict
    }

    if verbose: print(f"\nMetrics:\n" + json.dumps(metrics, indent=2))
    

    cm = confusion_matrix(y_true, y_pred, labels=range(dataset.n_classes))
    if verbose: print("confusion matrix\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(colorbar=True, cmap="Blues")
    tick_size = 8
    plt.xticks(rotation=30, ha='right', size=tick_size)
    plt.yticks(size=tick_size)
    label_size = 18
    if titleConfMatrix: plt.title(titleConfMatrix, fontsize=20, fontweight="bold")
    plt.xlabel("Predicted label", size=label_size)
    plt.ylabel("True label", size=label_size)
    plt.subplots_adjust(bottom=0.2, top=None if titleConfMatrix else 0.98)
    plt.tight_layout()

    with open(os.path.join(output_folder, "metrics.txt"), "w") as f:
        json.dump(metrics, f)
    plt.savefig(filename or os.path.join(output_folder, titleConfMatrix + '.png'))