import datetime
import time
import numpy as np
import os
import glob
import sys
import json
from functools import partial
import torch

from typing import List, Callable

from utils.utils1 import determine_dimensions_from_collection, file_to_tree_type_name


##############################
def preprocess_geojson_files(identifier: int, data_dir: str, what_happens_to_nan: str='keep_nan', bands_to_delete: List[str]=[], 
                             transformer_for_numpy_array: partial = None, transformers_data_augmentation: List[Callable|partial] = None,
                             fill_nan_value: float = None, continue_augmentation: bool = True, augment_rgb_only: bool = False,
                             verbose: bool=True):
    '''
    This function preprocesses the geojson files. The big goal is to create a numpy array for each sample and store them 
    accordingly in a dedicated folder. 

    identifier: number that identifies the way how the geojson file was created (e.g. 5x5 mask or 3x3 mask and so on)
                Only geojson-files with this identifier tag are preprocessed
    data_dir: folder where the geojson-files are stored
    what_happens_to_nan: preprocessing method of what happens to nan-values in the numpy array.
                         <keep_nan> DEFAULT:    all nan values are kept
                         <apply_nan_mask>:      for each band, a mask is generated (0 for nan, 1 for numeric) and 
                                                concatenated to the original array. NaNs are replaced by 'replace_nan'.
                                                Number of bands are doubled.
                         <delete_nan_samples>:  samples that contain nan are not written to disk
                         <replace_nan>:         NaN is replaced by 'fill_nan_value'
    bands_to_delete -> DEFAULT None: band names in this list are not considered when writting the arrays to disk. 
    transformer_for_numpy_array -> DEFAULT None: transformer function that transforms numpy array before
                                                 preprocessing method is applied (e.g. np.nanmean, np.nanmedian,...). 
    transformers_data_augmentation -> DEFAULT None: List of transformer functions from torchvision.transforms.v2.functional
                                                   (https://pytorch.org/vision/stable/transforms.html#transform-classes-functionals-and-kernels)
                                                   that used for data augmentation of the arrays. Not applied if <transformer_for_numpy_array> is not None.

    The arrays are written as .npy-files with the naming convention:
    "<label>-<index>-<name_of_transformer_data_augmentation>.npy"
    to disk in certain folders, whereas "-<name_of_transformer_data_augmentation>" is not added for no augmentation transforms. 
    The folders have the following naming convention:
    "<identifier>_<what_happens_to_nan>_<name_of_transformer_function>_<bands_to_delete>", 
    whereas ".<bands_to_delete>" is not added for no bands.
    '''

    # check if <what_happens_to_nan> argument contains valid element
    valid_preprocess_methods = ['keep_nan', 'apply_nan_mask', 'delete_nan_samples', 'replace_nan']
    if not what_happens_to_nan in valid_preprocess_methods:
        sys.exit(f"\n<{what_happens_to_nan}> not in list of valid preprocessing methods: {valid_preprocess_methods}. Terminating.")

    # check if <bands_to_delete> argument contains valid elements
    valid_bands = ['B11', 'B11_1', 'B11_2', 'B12', 'B12_1', 'B12_2', 'B2', 'B2_1', 'B2_2', 'B3', 'B3_1', 'B3_2', 'B4', 'B4_1', 'B4_2',
                   'B5', 'B5_1', 'B5_2', 'B6', 'B6_1', 'B6_2', 'B7', 'B7_1', 'B7_2', 'B8', 'B8A', 'B8A_1', 'B8A_2', 'B8_1', 'B8_2']
    if len(bands_to_delete) == 0:
        delete_bands_str = '' # later used for naming folder
    elif all([x in valid_bands for x in bands_to_delete]):
        delete_bands_str = '_' + '-'.join(bands_to_delete) # later used for naming folder
        valid_bands = [x for x in valid_bands if x not in bands_to_delete]
    else:
        sys.exit(f'\nSome given bands {bands_to_delete} are not part of the valid band names: {valid_bands}. Terminating.')

    # find all geojson files based on identifier
    search = os.path.join(data_dir, f"[A-Z]*_{identifier}.geojson")
    tree_type_files = glob.glob(search)
    if not tree_type_files:
        sys.exit(f"\nNo geojson-files found in folder {data_dir} or for identifier {identifier}. Terminating.")

    # find all tree type names
    tree_types = [file_to_tree_type_name(fn_, identifier) for fn_ in tree_type_files]
    if not tree_types:
        sys.exit(f"\nThe found geojson-files don't seem to match the standard naming convention \'<Tree>_<species>_<identifier>.geojson\'. Terminating.")
    
    # if not existent, create folder to save numpy arrays
    fill_nan_str = f"_fill_nan_{fill_nan_value}" if what_happens_to_nan in ["replace_nan", "apply_nan_mask"] else ""
    if transformer_for_numpy_array is None:
        transformer_name = ''
    else:
        transformer_name = '_' + transformer_for_numpy_array.func.__name__
    output_dir = os.path.join(data_dir, f'{identifier}_{what_happens_to_nan}' + fill_nan_str + f'{transformer_name}{delete_bands_str}')
    os.makedirs(output_dir, exist_ok = True)

    # statistical dictionary for output information
    sample_information = {}
    delete_information = {}

    # loop over all tree type names
    samples_augmented = 0
    tree_types_augmented = 0
    t0 = time.time()
    n_species = len(tree_types)
    for i_species, tree_type in enumerate(tree_types):

        # initialization of statistical dictionary for output information
        amount_of_samples = 0
        amount_of_samples_deleted = 0

        # open geojson-file
        file_name = os.path.join(data_dir, f'{tree_type}_{identifier}.geojson')
        with open(file_name) as f: data = json.load(f)

        # determine dimensions of data
        dimensions = determine_dimensions_from_collection(data)

        # loop over each sample
        n_features = len(data["features"])
        for s_, sample in enumerate(data["features"]):

            # output path of array
            output_path = os.path.join(output_dir, f'{tree_type}-{s_}')
            if continue_augmentation and os.path.exists(output_path+".npy"): continue

            # create numpy array for sample
            array, remaining_bands = sample2numpy(sample, bands_to_delete, *dimensions)

            # transformer with partial is applied to the array (e.g. np.nanmean)
            if transformer_for_numpy_array is not None:
                try:
                    array = transformer_for_numpy_array(array)
                except Exception as err:
                    sys.exit(f"\nThe transformer threw an unexpected error {type(err)}: {err}. Terminating.")
            
            # samples containing nan are not written to disk
            if (what_happens_to_nan == 'delete_nan_samples') and (np.isnan(array).any()):
                amount_of_samples_deleted += 1 # counter for samples
                continue
            
            # nan mask is concatenated to numpy array (0 for nan, 1 for numeric value)
            # channel dimension is doubled
            if what_happens_to_nan == 'apply_nan_mask':
                assert fill_nan_value is not None, "If what_happens_to_nan = 'apply_nan_mask', a value for fill_nan_value has to be provided to replace the NaNs"
                mask = (~np.isnan(array)).astype(array.dtype)
                array[np.isnan(array)] = fill_nan_value
                array = np.concatenate((array, mask), axis=0) #TODO: change numpy array dimensions from (h,w,b) to (b,h,w) for less confusing array visualization 

            if what_happens_to_nan == 'replace_nan':
                assert fill_nan_value is not None, "If what_happens_to_nan = 'replace_nan', a value for fill_nan_value has to be provided to replace the NaNs"
                array[np.isnan(array)] = fill_nan_value

            # array is saved as .npy-file in dedicated folder
            amount_of_samples += 1 # counter for samples

            # save
            assert not np.isnan(array).any(), "NaN before saving:" + output_path
            np.save(output_path+'.npy', array, allow_pickle=False)

            # data augmentation
            if (transformers_data_augmentation is not None) and (transformer_for_numpy_array is None):
                data_augmentation(array, transformers_data_augmentation, output_path, remaining_bands, augment_rgb_only)
                amount_of_samples += len(transformers_data_augmentation)

            samples_augmented += 1
            t_per_augm = (time.time()-t0)/samples_augmented
            if s_%40 == 0: print(f"Sample {s_:4d}/{n_features} ({s_/n_features:.2f})\tt/augm {t_per_augm:.2f}\tETA (this species): {datetime.timedelta(seconds=(n_features-(s_+1))*t_per_augm)}\tspecies [{i_species}/{n_species}]")
        tree_types_augmented += 1

        # fill statistical dictionary for output information
        sample_information[tree_type] = amount_of_samples
        delete_information[tree_type] = amount_of_samples_deleted
        
        if verbose:
            print(f'{"<"+tree_type+">":<30} {amount_of_samples} samples written to disk.')


    preprocessing_info = \
    f'\nIdentifier: {identifier}' + \
    f'\nChosen processing method: {what_happens_to_nan}' + \
    f'\nNot considered bands: {bands_to_delete}' + \
    f'\nTransformer: {transformer_for_numpy_array}' + \
    f'\nAugmentation Transformer: {transformers_data_augmentation}' + \
    f'\nTree types considered: {tree_types}' + \
    f'\nAmount of samples written:\n {json.dumps(sample_information, indent=2)}' + \
    f'\nAmount of samples deleted:\n {json.dumps(delete_information, indent=2)}\n'
    with open(os.path.join(output_dir, "preprocessing_info.txt"), "w") as f:
        f.write(preprocessing_info)
    if verbose: print(preprocessing_info)
            


def sample2numpy(sample: dict, bands_to_delete: List[str], w: int=25, h: int=25, b: int=30) -> np.array:
    '''
    This function converts the geojson strcture (dicctionary) to a numpy array
    axis = 0: channels
    axis = 1: height
    axis = 2: width

    sample: dictionary structure of the geojson file
    bands_to_delete: band names that should not be written to disk
    '''
    b -= len(bands_to_delete)

    # delete bands
    properties = sample["properties"]
    for key in bands_to_delete:
        del properties[key]

    # fill up array
    array = np.full((b,h,w), np.nan)
    for b_, band in enumerate(properties.values()):
        if band is None: continue       
        for r_, row in enumerate(band):
            array[b_, :, r_] = row
    remaining_bands = list(properties.keys())
    return array.astype(np.float32), remaining_bands



def data_augmentation(array: np.array, transforms: List[Callable|partial], output_path: str,
                      remaining_bands: List[str], augment_rgb_only: bool = False):
    '''
    This function performs data augmentation on the given array based on a list of transforms. For
    each given transform, a new npy file is written with the naming convention:
    "<output_path>-<name_of_transform>.npy"

    array: array that should be augmented
    transforms: List of transformer functions from torchvision.transforms.v2.functional
                (https://pytorch.org/vision/stable/transforms.html#transform-classes-functionals-and-kernels)
                that are used for data augmentation of the arrays
    output_path: output path of array without ".npy" at the end
    '''
    transforms = list(set(transforms)) # only use unique transforms
    # loop over all transforms
    for transform in transforms:
        try:
            transf_array = transform(torch.from_numpy(array)).numpy() # apply transform
        except TypeError as e:
            if "Input image tensor permitted channel values are 1 or 3" in e.args[0] or "Input image tensor can have 1 or 3 channels" in e.args[0]:
                transf_array = np.zeros_like(array)
                if augment_rgb_only:
                    rgb_ids = ["B4", "B3", "B2"]
                    subscripts = np.unique([b_.split("_")[-1] if "_" in b_ else "" for b_ in remaining_bands])
                    for subscript_ in subscripts:
                        for rgb_id_ in rgb_ids:
                            subscript_str = f"_{subscript_}" if subscript_ else ""
                            combined_id = f"{rgb_id_}{subscript_str}"
                            print("combined id:", combined_id)
                            if combined_id not in remaining_bands: raise ValueError(f"Band ID '{combined_id}' has to be in remaining bands if augment_rgb_only=True")
                else:
                    n_sub = array.shape[0] // 3     # determine how many "RGBs" can be created
                    indices = np.arange(array.shape[0])
                    np.random.shuffle(indices)      # shuffle to avoid always cutting of the last
                    indices = indices[:n_sub*3]     # potential left over cut off
                    for idxs in zip(indices[::3], indices[1::3], indices[2::3]):
                        transf_array[idxs, :, :] = transform(torch.from_numpy(array[idxs, :, :])).numpy()
            else:
                raise e

        if isinstance(transform, partial):
            transform_name = transform.func.__name__ # get name of function
            # add name and value of function keywords
            if transform.keywords:
                for key, value in transform.keywords.items():
                    transform_name += '-' + key + '=' + str(value)
        else:
          transform_name = transform.__name__
        # save numpy file
        np.save(f'{output_path}-{transform_name}.npy', transf_array, allow_pickle=False)

##############################


if __name__ == "__main__":
    # identifier = "test"
    # data_dir = 'data/test'

    identifier = "train_val"
    data_dir = "data/train_val"

    # bands_to_delete = ['B11_2', 'B12_2', 'B2_2', 'B3_2', 'B4_2', 'B5_2', 'B6_2', 'B7_2', 'B8A_2', 'B8_2']
    bands_to_delete = []

    # PREPROCESSING OPTIONS ##################################################################################
    # what_happens_to_nan='apply_nan_mask'
    # what_happens_to_nan='replace_nan'
    what_happens_to_nan='delete_nan_samples'
    # what_happens_to_nan='keep_nan'
    nan_replace = -1

    # TRANSFORMER ############################################################################################
    # either use transformer_for_numpy_array or transformers_data_augmentation

    # transformer_for_numpy_array
    transformer = None
    #transformer = partial(np.nanmean, axis=(1,2))  # mean per band excluding nan values
    #transformer = partial(np.nanmedian, axis=(1,2)) # median per band excluding nan values

    # transformers_data_augmentation
    # see https://pytorch.org/vision/stable/transforms.html#transform-classes-functionals-and-kernels
    # only use functional functions!
    from torchvision.transforms.v2 import functional
    data_aug_transformers = [
        partial(functional.adjust_brightness, brightness_factor=0.5),
        partial(functional.adjust_brightness, brightness_factor=2),
        partial(functional.adjust_contrast, contrast_factor=.5),
        partial(functional.adjust_contrast, contrast_factor=2),
        partial(functional.adjust_hue, hue_factor=-.3),
        partial(functional.adjust_hue, hue_factor=.3),
        partial(functional.adjust_saturation, saturation_factor=0.5),
        partial(functional.adjust_saturation, saturation_factor=2),
        partial(functional.adjust_sharpness, sharpness_factor=0.5),
        partial(functional.adjust_sharpness, sharpness_factor=1),
        functional.autocontrast,
        partial(functional.gaussian_blur, kernel_size=3, sigma=0.2),
        partial(functional.rotate, angle=90),
        partial(functional.rotate, angle=180),
        partial(functional.rotate, angle=270),
        functional.horizontal_flip,
        functional.vertical_flip,
        ]

    # PREPROCESSING ########################################################################################
    outdir = preprocess_geojson_files(identifier, data_dir, what_happens_to_nan, bands_to_delete,
                                      transformer_for_numpy_array=transformer, transformers_data_augmentation=data_aug_transformers, fill_nan_value=nan_replace,
                                      continue_augmentation=False)

    # Checking one sample
    #arr = np.load(r'data/1102_delete_nan_samples_B2/Abies_alba-21.npy')
    #arr2 = np.load(r'data/1102_delete_nan_samples_B2/Abies_alba-21-rotate-angle=180.npy')
    #print(arr[0,:,:])
    #print(arr2[0,:,:])
    #print(arr.shape)
    #print(arr2.shape)
