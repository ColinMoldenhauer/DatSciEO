import json
import os
import glob
import re
import sys

from typing import Iterable
from matplotlib.container import BarContainer

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Dataset

from utils import sample_file_to_tree_type, determine_dimensions_from_collection


# TODO: get ID and geometry in export
# TODO: reshape data?
# TODO: keep all features in memory? or read from file everytime? -> quite slow, replace with in memory? or preprocessing
# TODO: how to deal with class imbalance? -> Dataloader task?
class TreeClassifDataset(Dataset):
    def __init__(self, data_dir, identifier, verbose=False, *args, **kwargs):
        """
        A dataset class for the Tree Classification task.

        :param data_dir: The directory where to find the geojson files.
        :type data_dir: str
        :param identifier: The identifier to filter dataset files.
        :type identifier: str
        :param verbose: whether to print on instantiation, defaults to True
        :type verbose: bool, optional
        """

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.identifier = identifier

        # find files based on identifier
        search = os.path.join(data_dir, f"[A-Z]*_{identifier}.geojson")
        self.class_files = glob.glob(search)

        # collect information about the dataset and build an index
        self.classes = [self.file_to_classname(fn_) for fn_ in self.class_files]
        self.samples_per_class = {cn_: self.count_samples(fn_) for cn_, fn_ in zip(self.classes, self.class_files)}
        self.cumulative = np.cumsum(list(self.samples_per_class.values()))
        self._create_band_index(self.class_files[0])
        self.determine_dimensions()

        if verbose: print(str(self))

    def __len__(self): return sum(list(self.samples_per_class.values()))

    def __getitem__(self, index, verbose=False):
        if index >= len(self): raise IndexError(f"Index {index} too large for dataset of length {len(self)}.")
        
        # determine which file to open
        file_idx, file = self.index_to_file(index, return_index=True)

        # determine which feature to load from the file
        idx_offsets = np.roll(self.cumulative, 1)
        idx_offsets[0] = 0
        rel_idx = index - idx_offsets[file_idx]

        # load feature
        with open(file) as f: collection = json.load(f)
        feature = collection["features"][rel_idx]

        # transform into numpy array
        data = self.feature_to_numpy(feature)
        
        if verbose:
            print("idx:", index, "file idx:", file_idx, "file:", file, "rel idx", rel_idx)
            print("selected feature:", feature["properties"]["B11"][0])

        # return data as a numpy array and the class label, which is equal to file_idx
        return data, file_idx

    def __str__(self):
        s = f"Dataset found for identifier '{self.identifier}':\n\t" + "\n\t".join(self.class_files)
        s += f"\n  -> Classes ({len(self.classes)}):"
        for c, n in self.samples_per_class.items(): s+= f"\n\t{c:<20} {n:>5} samples"
        s += f"\n  -> Samples: {len(self)}"
        return s


    def determine_dimensions(self):
        w, h, b = None, None, None
        for f_ in self.class_files:
            with open(f_) as f: collection = json.load(f)
            w, h, b = determine_dimensions_from_collection(collection)
            if w is not None: break
        
        if w is None:
            raise ValueError("Dataset seems to be empty.")
        else:
            self.width = w
            self.height = h
            self.depth = b
    

    def feature_to_numpy(self, feature):
        b = self.depth
        h = self.height
        w = self.width

        data = np.full((h, w, b), np.nan)
        for b_, band in enumerate(feature["properties"].values()):
            if band is None: continue       # TODO: how to handle NaN?
            for r_, row in enumerate(band):
                data[r_, :, b_] = row
        return data


    def index_to_file(self, index, return_index=False):
        larger = self.cumulative > index
        file_idx = np.argmax(larger)
        file = self.class_files[file_idx]

        return (file_idx, file) if return_index else file

    def file_to_classname(self, filename):
        classname = re.search(f"([A-Z][a-z]+_[a-z]+)_{self.identifier}.geojson", filename).group(1)
        return classname
    
    def index_to_classname(self, index):
        file = self.index_to_file(index)
        return self.file_to_classname(file)

    def label_to_classname(self, label):
        return self.file_to_classname(self.class_files[label])

    def _create_band_index(self, file):
        with open(file) as f: collection = json.load(f)
        feature = collection["features"][0]
        self.bands = list(feature["properties"].keys())
        self.band2index = {band_: i for (i, band_) in enumerate(self.bands)}
        self.index2band = {i: band_ for (i, band_) in enumerate(self.bands)}

    def count_samples(self, filename):
        with open(filename) as f:
            data = json.load(f)
        return len(data["features"])


    def visualize_samples(self, indices, subplots, band_names=["B5", "B4", "B3"], **kwargs):
        fig, axs = plt.subplots(*subplots, **kwargs)
        fig.suptitle(f"Tree Samples (bands {list(band_names)})")
        axs = axs.flatten() if len(indices) != 1 else [axs]

        band_indices = [self.band2index[b_] for b_ in band_names]

        for i_, ax_ in zip(indices, axs):
            data, label = self[i_]
            ax_.imshow(data[:, :, band_indices])
            ax_.set_title(f"{i_}: {self.label_to_classname(label)}")
        
        return fig
    
    def band_nan_histogram(self, normalize_nan_sum=True, individual_plots=True, show=False, savedir=""):
        """
        Visualizes the dataset's data availability.

        - top subplot: number of samples per class and band that are missing
        - middle subplot: histogram of how many bands are missing per sample
        - bottom subplots: histograms of NaN occurance per layer (for our case obsolete)
        """
        if savedir: os.makedirs(savedir, exist_ok=True)
        # collect some statistics
        empty_bands_per_class = []
        hist_all = []
        nan_all = {}
        nan_samples = []
        for f_ in self.class_files:
            empty_bands_per_feature = []
            nan_samples_per_class = 0
            nan_per_class = np.zeros(self.depth)
            nan_concat = None
            with open(f_) as f: collection = json.load(f)
            for feature_ in collection["features"]:
                data = self.feature_to_numpy(feature_)
                nan_per_band = np.isnan(data).sum(axis=(0,1))
                if nan_per_band.sum() > 0: nan_samples_per_class += 1
                # empty_bands = np.isnan(data).sum()
                empty_bands = np.isnan(data).sum()/self.width/self.height   # normalize by amount of pixels per layer
                # sum nan: either 0, 250 (one season missing), 750 (all seasons missing)
                nan_per_class += nan_per_band

                if nan_concat is None:
                    nan_concat = nan_per_band
                else:
                    nan_concat = np.vstack((nan_concat, nan_per_band))

                empty_bands_per_feature.append(empty_bands)

            empty_bands_per_class.append(empty_bands_per_feature)
            nan_all[self.file_to_classname(f_)] = nan_per_class
            nan_samples.append(nan_samples_per_class)
            hist_all.append(nan_concat)

        # initialize subplots
        if individual_plots:
            # plot first plot

            keys = np.array(list(nan_all.keys()))   # split into multiple for better readability
            for i, classes_ in enumerate(np.array_split(keys, len(nan_all)//3)):
                nan_all_ = {c:nan_all[c] for c in classes_}
                plt.figure(figsize=(12, 8))
                x = np.arange(self.depth)  # the label locations
                width = 0.4  # the width of the bars
                multiplier = 0
                norm = self.width*self.height if normalize_nan_sum else 1

                for attribute, measurement in nan_all_.items():
                    offset = width * multiplier
                    rects = plt.bar(x + offset, measurement/norm, width, label=attribute, align="edge")
                    plt.bar_label(rects, padding=3, rotation=90, fmt=lambda x: str(int(x)) if x else "")
                    multiplier += 1
                
                
                plt.title("Total NaNs per band per class")
                plt.xlabel("Bands")
                plt.ylabel("Count", labelpad=15)
                plt.xticks(x+width*len(nan_all_)/2, self.bands, rotation=30, fontsize="medium")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize="x-large", fancybox=True, shadow=False, ncol=len(nan_all_))
                ylim = plt.gca().get_ylim()
                plt.ylim(ylim[0], 1.2*ylim[1])
                plt.tight_layout()
                if savedir: plt.savefig(os.path.join(savedir, f"plot_1_{i}.png"))
            

            # plot second plot
            plt.figure(figsize=(12, 8))
            plt.title("Frequency of missing bands")
            rwidth = 10.
            hist = plt.hist(empty_bands_per_class, bins=self.depth, align="mid", label=self.classes, rwidth=rwidth)
            for bar_container_ in hist[-1]:
                bar_container_nonzero = self._filter_empty_bars(bar_container_, upper=10)
                plt.bar_label(bar_container_nonzero, padding=3, rotation=0)
            
            plt.xlabel("NaN bands per sample")
            plt.ylabel("Count", labelpad=15)
            plt.legend()

            x0, x1 = plt.gca().get_xlim()
            visible_ticks = [int(t) for t in plt.xticks()[0] if t>=x0 and t<=x1]
            plt.xticks(visible_ticks + hist[2][0][0].get_width()*len(self.classes)/2, visible_ticks)

            plt.tight_layout()
            if savedir: plt.savefig(os.path.join(savedir, "plot_2.png"))

            # plot third plot
            for i, nan_class_ in enumerate(hist_all):
                fig = plt.figure(figsize=(20, 10))
                plt.hist(nan_class_)
                plt.title(f"Frequency of NaN pixels per band ({self.classes[i]})")
                plt.xlabel("NaN pixels per band")
                plt.ylabel("Count", labelpad=15)
                plt.tight_layout()
                if savedir: plt.savefig(os.path.join(savedir, f"plot_3_{i}.png"))
            
            # plot fourth plot: class distribution
            fig = plt.figure(figsize=(20, 10))
            plt.title("Class Distribution")
            plt.xlabel("Species")
            plt.ylabel("Count", labelpad=15)
            plt.xticks(np.arange(len(self.classes)), self.classes, rotation=45)
            plt.bar(np.arange(len(self.classes)), list(self.samples_per_class.values()))
            plt.tight_layout()
            if savedir: plt.savefig(os.path.join(savedir, f"plot_class_distr.png"))

            # plot fifth plot: NaN per class
            fig = plt.figure(figsize=(20, 10))
            plt.title("Class Distribution")
            plt.xlabel("Species")
            plt.ylabel("Count", labelpad=15)
            width = .2
            n_bars = 2
            x = np.arange(len(self.classes))
            n_samples = list(self.samples_per_class.values())
            plt.bar(x, n_samples, width)
            plt.bar(x+width, np.array(n_samples) - np.array(nan_samples), width)
            plt.bar(x+2*width, nan_samples, width)
            plt.xticks(x+width*n_bars/2, self.classes, rotation=45)
            plt.tight_layout()
            plt.legend(["Samples", "Samples filtered", "Samples with NaN"], fontsize="large")
            if savedir: plt.savefig(os.path.join(savedir, f"plot_class_distr_all.png"))

            
            # plot fourth plot: class distribution filtered
            fig = plt.figure(figsize=(20, 10))
            plt.title("Class Distribution (filtered)")
            plt.xlabel("Species")
            plt.ylabel("Count", labelpad=15)
            plt.xticks(np.arange(len(self.classes)), self.classes, rotation=45)
            plt.bar(np.arange(len(self.classes)), np.array(list(self.samples_per_class.values()))-np.array(nan_samples))
            plt.tight_layout()
            if savedir: plt.savefig(os.path.join(savedir, f"plot_class_distr_filt.png"))
            
            
        else:
            fig = plt.figure(figsize=(20, 10))
            ax_bar1 = fig.add_subplot(3, 1, 1)
            ax_bar2 = fig.add_subplot(3, 1, 2)
            axs_hist = [fig.add_subplot(3, 3, i) for i in range(7, 7+len(hist_all))]

            # plot first plot
            x = np.arange(self.depth)  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            norm = self.width*self.height if normalize_nan_sum else 1

            for attribute, measurement in nan_all.items():
                offset = width * multiplier
                rects = ax_bar1.bar(x + offset, measurement/norm, width, label=attribute, align="edge")
                ax_bar1.bar_label(rects, padding=3, rotation=90)
                multiplier += 1
            
            ax_bar1.set_title("Total NaNs per band per class")
            ax_bar1.set_xticks(x+width*len(self.classes)/2, self.bands, rotation=90)
            ax_bar1.legend(loc='upper left')
            ylim = ax_bar1.get_ylim()
            ax_bar1.set_ylim(ylim[0], 1.2*ylim[1])
            

            # plot second plot
            hist = ax_bar2.hist(empty_bands_per_class, bins=self.depth, align="mid")
            for bar_container_ in hist[-1]:
                bar_container_nonzero = self._filter_empty_bars(bar_container_)
                ax_bar2.bar_label(bar_container_nonzero, padding=3, rotation=90)
            
            ax_bar2.set_xlabel("NaNs per sample")
            ax_bar2.set_ylabel("Samples", labelpad=15)

            # plot third subplots
            for i, (ax_, nan_class_) in enumerate(zip(axs_hist, hist_all)):
                ax_.hist(nan_class_)
                ax_.legend(title=self.classes[i])
                ax_.set_xlabel("NaNs per band")
                ax_.set_ylabel("Samples", labelpad=15)

            plt.tight_layout(pad=3, h_pad=.4)

        if show: plt.show()

    def _filter_empty_bars(self, bar_container, upper=None, lower=0):
        rects = np.array([rect for rect in bar_container])
        datavalues = bar_container.datavalues
        heights = np.array([rect.get_height() for rect in rects])
        upper = upper or heights.max()
        mask_upper = heights < upper
        mask_lower = heights > lower
        mask = mask_upper & mask_lower

        rects_nonzero = rects[mask]
        datavals_nonzero = datavalues[mask]
        bar_container_nonzero = BarContainer(rects_nonzero, datavalues=datavals_nonzero, orientation="vertical")
        return bar_container_nonzero


# TODO: add band information?
class TreeClassifPreprocessedDataset(Dataset):
    def __init__(self, data_dir: str, torchify: bool = False, indices: Iterable = None,
                 ignore_augments: Iterable[str] = [], excludeAugmentationFor: Iterable[str] = []):
        """
        A dataset class for the Tree Classification task.
        Samples need to be created using preprocessing.preprocess_geojson_files() first.

        :param data_dir: Path to the preprocessed data directory
        :param torchify: Whether to flip the data dimensions for torch. Will be removed soon, after Chris
                            implements the flipping per default in the preprocessing.
        :param indices: Optional array indices to load a subset of the dataset; useful for testing purposes
        :param ignore_augments: Iterable of strings, which augmentations to ignore
        :param excludeAugmentationFor: Iterable of tree species, for which augmentations to ignore
        
        """
        super().__init__()
        self.data_dir = data_dir
        self.torchify = torchify

        if ignore_augments == "all":
            self.files = [file_ for file_ in os.listdir(data_dir) if re.match("[A-Z][a-z]+_[a-z]+-\d+[.]npy", file_)]
        else:
            self.files = [file_ for file_ in os.listdir(data_dir) if (file_.endswith(".npy") and not any([ign_ in file_ for ign_ in ignore_augments]))]


        # exclude augmented data for given tree species
        if excludeAugmentationFor:
            classes = list(np.unique([sample_file_to_tree_type(file_) for file_ in self.files]))
            for species in excludeAugmentationFor:
                if species in classes:
                    delete_from_files = [file_ for file_ in self.files if (file_.startswith(species+'-')) & (len(file_.split('-')) > 2)]
                    if delete_from_files:
                        self.files = [file_ for file_ in self.files if file_ not in delete_from_files]
                else:
                    sys.exit(f'The tree species {species} for which you want to exclude augmented data, does not exist. Terminating.')


        if indices: self.files = [self.files[idx] for idx in indices]

        self.classes = list(np.sort(np.unique([sample_file_to_tree_type(file_) for file_ in os.listdir(data_dir) if file_.endswith(".npy")])))
        self.samples_per_class = {cl_: len([True for file_ in self.files if re.match(cl_, file_) ]) for cl_ in self.classes}
        self._set_dimensions()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = np.load(os.path.join(self.data_dir, self.files[index]))
        tree_type = sample_file_to_tree_type(self.files[index])
        class_idx = self.labelname_to_label(tree_type)
        
        if self.torchify:
            # size as saved by preprocessing: W, H, C
            # size as required by torch: C, H, W
            data = np.moveaxis(data, [0,1,2], [2,1,0])
        return data, class_idx
    
    def __str__(self):
        n_samples = len(self)
        s = f"TreeClassifPreprocessedDataset:\n"
        s += f"  -> Classes ({len(self.classes)}):"
        for c, n in self.samples_per_class.items(): s+= f"\n\t{c:<20} {n:>5} samples\t( {n/n_samples:2.0%} )"
        s += f"\n  -> Samples: {n_samples}"
        return s

    def _set_dimensions(self):
        x = np.load(os.path.join(self.data_dir, self.files[0]))
        if self.torchify:
            self.width, self.height, self.depth = x.shape
        else:
            self.depth, self.width, self.height = x.shape
        self.n_classes = len(self.classes)
    
    def labelname_to_label(self, labelname):
        return self.classes.index(labelname)
    
    def label_to_labelname(self, label):
        return self.classes[label]
        
    def visualize_samples(self, indices, subplots, band_names=["B5", "B4", "B3"], **kwargs):
        fig, axs = plt.subplots(*subplots, **kwargs)
        fig.suptitle(f"Tree Samples (bands {list(band_names)})")
        axs = axs.flatten() if len(indices) != 1 else [axs]

        band_indices = [self.band2index[b_] for b_ in band_names]

        for i_, ax_ in zip(indices, axs):
            data, label = self[i_]
            ax_.imshow(data[:, :, band_indices])
            ax_.set_title(f"{i_}: {self.label_to_classname(label)}")
        
        return fig

    def band_nan_histogram(self, normalize_nan_sum=True):
        """
        Visualizes the dataset's data availability.

        - top subplot: number of samples per class and band that are missing
        - middle subplot: histogram of how many bands are missing per sample
        - bottom subplots: histograms of NaN occurance per layer (for our case obsolete)
        """
        # collect some statistics
        empty_bands_per_class = []
        hist_all = []
        nan_all = {}
        for class_ in self.classes:
            indices_per_class_ = [i for i, f_ in enumerate(self.files) if f_.startswith(class_)]
            empty_bands_per_feature = []
            nan_per_class = np.zeros(self.depth)
            nan_concat = None
            for idx_ in indices_per_class_:
                data = self[idx_][0]
                print("data:", data.shape)
                print("depth:", self.depth)
                nan_per_band = np.isnan(data).sum(axis=(0,1))
                empty_bands = np.isnan(data).sum()/self.width/self.height
                # sum nan: either 0, 250 (one season missing), 750 (all seasons missing)
                nan_per_class += nan_per_band

                if nan_concat is None:
                    nan_concat = nan_per_band
                else:
                    nan_concat = np.vstack((nan_concat, nan_per_band))

                empty_bands_per_feature.append(empty_bands)

            empty_bands_per_class.append(empty_bands_per_feature)
            nan_all[class_] = nan_per_class
            hist_all.append(nan_concat)

        # initialize subplots
        fig = plt.figure(figsize=(20, 10))
        ax_bar1 = fig.add_subplot(3, 1, 1)
        ax_bar2 = fig.add_subplot(3, 1, 2)
        axs_hist = [fig.add_subplot(3, 3, i) for i in range(7, 10)]

        # plot first plot
        x = np.arange(self.depth)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        norm = self.width*self.height if normalize_nan_sum else 1

        for attribute, measurement in nan_all.items():
            offset = width * multiplier
            rects = ax_bar1.bar(x + offset, measurement/norm, width, label=attribute, align="edge")
            ax_bar1.bar_label(rects, padding=3, rotation=90)
            multiplier += 1
        
        ax_bar1.set_title("Total NaNs per band per class")
        ax_bar1.set_xticks(x+width*len(self.classes)/2, self.bands, rotation=90)
        ax_bar1.legend(loc='upper left')
        ylim = ax_bar1.get_ylim()
        ax_bar1.set_ylim(ylim[0], 1.2*ylim[1])
        

        # plot second plot
        hist = ax_bar2.hist(empty_bands_per_class, bins=self.depth, align="mid")
        for bar_container_ in hist[-1]:
            bar_container_nonzero = self._filter_empty_bars(bar_container_)
            ax_bar2.bar_label(bar_container_nonzero, padding=3, rotation=90)
        
        ax_bar2.set_xlabel("NaNs per sample")
        ax_bar2.set_ylabel("Samples", labelpad=15)


        # plot third subplots
        for i, (ax_, nan_class_) in enumerate(zip(axs_hist, hist_all)):
            ax_.hist(nan_class_)
            # ax_.set_title(self.classes[i])
            ax_.legend(title=self.classes[i])
            ax_.set_xlabel("NaNs per band")
            ax_.set_ylabel("Samples", labelpad=15)

        plt.tight_layout(pad=3, h_pad=.4)
        plt.show()

    def _filter_empty_bars(self, bar_container):
        rects = np.array([rect for rect in bar_container])
        datavalues = bar_container.datavalues
        heights = np.array([rect.get_height() for rect in rects])
        mask_nonzero = heights > 0

        rects_nonzero = rects[mask_nonzero]
        datavals_nonzero = datavalues[mask_nonzero]
        bar_container_nonzero = BarContainer(rects_nonzero, datavalues=datavals_nonzero, orientation="vertical")
        return bar_container_nonzero


class TorchStandardScaler:
  def fit(self, x):
    self.mean = x.mean(0, keepdim=True)
    self.std = x.std(0, unbiased=False, keepdim=True)

  def transform(self, x):
    return (x - self.mean) / (self.std + 1e-7)

  def save(self, path):
     torch.save({
        "mean": self.mean,
        "std": self.std
        }, path)

  def load(self, path):
     params = torch.load(path)
     self.mean = params["mean"]
     self.std = params["std"]


# test dataset
if __name__ == "__main__":
    # test initialization
    # ds = TreeClassifDataset("data", "1102")
    # print("len ds:", len(ds))
    
    # test getitem
    # sample00, label00 = ds[0]
    # print(sample00, sample00.shape, sep="\n")

    # sample0l, label0l =  ds[980]
    # sample10, label10 =  ds[981]
    # sample1l, label1l =  ds[4742]
    # sample20, label20 =  ds[4743]
    # sample2l, label2l =  ds[4747]
    # sampleerror, labelerror = ds[4748]

    # test visualization
    # fig = ds.visualize_samples(np.random.randint(0, len(ds)-1, 12), (3,4))
    # plt.show()

    # test histogram
    # ds.band_nan_histogram()

    dsp = TreeClassifPreprocessedDataset("data/1102_delete_nan_samples_B2", excludeAugmentationFor=['Picea_abies'], ignore_augments=['vertical_flip'])
    x0, y0 = dsp[0]
    print("dataset shape:", len(dsp))
    print("dataset sample:", type(dsp[0]))
    print("data shape:", x0.shape)
    print("label:", y0)
    print("labelname:", dsp.label_to_labelname(y0))
