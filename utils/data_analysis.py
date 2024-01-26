import os
from matplotlib import pyplot as plt
import numpy as np
import torch

from dataset import TreeClassifDataset, TreeClassifPreprocessedDataset, TorchStandardScaler


def plot_data_histogram(x, bins=20, normalize=True, bands_per_plot=None, title="", **kwargs):
   n_plots = x.shape[1] / bands_per_plot if bands_per_plot else 1
   x_per_band = x.reshape((-1, x.shape[1]))

   if normalize:
      weights = np.full_like(x_per_band, 1./x_per_band.shape[0])
   else:
      weights = np.full_like(x_per_band, 1)
    
   x_split = np.array_split(x_per_band, n_plots, axis=1)
   for i, x_ in enumerate(x_split):
    plt.figure(**kwargs)
    plt.title(title)
    plt.hist(x_, bins, weights=weights)
    plt.xlabel("Pixel Values")
    plt.ylabel("Percentage" if normalize else "Count", labelpad=15)


if __name__ == "__main__":
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    # plt.rcParams['xtick.major.pad'] = 12
    # plt.rcParams['ytick.major.pad'] = 12


    # dataset_dir = "data/1123_top10"
    # ds = TreeClassifDataset(dataset_dir, 1123)
    # ds.band_nan_histogram(savedir=os.path.join(dataset_dir, "data_analysis"))

    dataset_dir = "data/1123_top10/1123_delete_nan_samples"
    dsp = TreeClassifPreprocessedDataset(dataset_dir, indices=range(200))
    x = torch.tensor(np.array([x[0] for x in dsp]))
    scaler = TorchStandardScaler()
    scaler.fit(x)
    x_stand = scaler.transform(x)

    figure_dir = os.path.join(dataset_dir, "data analysis")
    os.makedirs(figure_dir, exist_ok=True)
    plot_data_histogram(x.numpy(), title="Distribution of Original Data", figsize=(12, 8))
    plt.savefig(os.path.join(figure_dir, "data_distribution_original.png"), bbox_inches='tight')
    plot_data_histogram(x_stand.numpy(), title="Distribution of Standardized Data", figsize=(12, 8))
    plt.savefig(os.path.join(figure_dir, "data_distribution_standardized.png"), bbox_inches='tight')
    # plt.show()