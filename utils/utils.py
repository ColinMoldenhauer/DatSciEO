import re


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
