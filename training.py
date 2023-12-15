# A simple training script for torch networks

import os
import time

import numpy as np

from models import TreeClassifConvNet
from utils import TreeClassifPreprocessedDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

# TODO: move to Google Colab -> check path behavior

###########################################################
##################      SETTINGS        ###################
###########################################################

# general training settings
N_epochs = 400
batch_size = 18
learning_rate = 1e-4
verbose = True


run_dir = f"runs/{time.strftime('%Y%m%d-%Hh%Mm%Ss', time.localtime())}"
checkpoint_dir = os.path.join(run_dir, "checkpoints")

os.makedirs(run_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# create datasets and dataloaders
dataset_dir = "data/1123_top10/1123_delete_nan_samples_"
splits = [.7, .3]
dataset = TreeClassifPreprocessedDataset(
    dataset_dir,
    torchify=True,
    indices=[
        *range(50),
        # *range(500, 550)
        ]
    )

generator = torch.Generator().manual_seed(42)
ds_train, ds_val = random_split(dataset, splits, generator=generator)

# debug datasets
ds_train = TreeClassifPreprocessedDataset(
    dataset_dir,
    torchify=True,
    indices=[
        *range(50),
        # *range(500, 550)
        ]
    )

n_samples = len(os.listdir(dataset_dir))
ds_val = TreeClassifPreprocessedDataset(
    dataset_dir,
    torchify=True,
    indices=[
        # *range(50),
        *range(n_samples-42, n_samples)
        ]
    )

# TODO: validate class balance after splits
# TODO: introduce sampler?
dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_val = DataLoader(ds_val, batch_size, shuffle=True)


# model, loss and optimizer
model = TreeClassifConvNet(
    n_classes = dataset.n_classes,
    width = dataset.width,
    height = dataset.height,
    depth = dataset.depth
)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

##################      SETTINGS END        ###################
###############################################################




if verbose: print(
    f"\nUsing dataset with properties:\n"       \
    f"\tsamples: {len(dataset)}\n"              \
    f"\t\ttrain: {len(ds_train)}\n"             \
    f"\t\tval:   {len(ds_val)}\n"               \
    f"\tshape: {dataset[0][0].shape}\n"         \
    )


# training loop
t0 = time.time()
last_info = t0
n_train = len(dl_train.dataset)
n_val = len(dl_val.dataset)

writer = SummaryWriter(run_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for i_epoch in range(N_epochs):

    running_loss = 0.
    last_loss = 0.
    correct = 0

    model.train()
    for i_batch, (x, y) in enumerate(dl_train):
        assert torch.isnan(x).sum() == 0, "NaN in trainings data, please fix."

        x = x.float().to(device)
        pred = model(x)

        loss = loss_fn(pred, y)
        running_loss += loss.item()
        loss.backward()

        optim.step()
        optim.zero_grad()

        if verbose and (((time.time() - last_info) > 20) or (i_batch % (n_train//n_train) == 0)):
            # TODO: ETA, avg epoch duration, etc...
            loss_avg = running_loss / (i_batch+1) # loss per batch
            last_info = time.time()
            loss, current = loss.item(), (i_batch + 1) * len(x)
            print(f"train:  {loss:>7f}  loss avg: {loss_avg:>7f}   [epoch {i_epoch:>3d}: {current:>5d}/{n_train:>5d}]")
    
    # TODO: do this always? could be slow for large datasets
    all = [(el, y_) for x, y in dl_train for el, y_ in zip(np.argmax(model(x.float().to(device)).detach().numpy(), axis=1), y.numpy())]
    all_preds = [a[0] for a in all]
    all_gts = [a[1] for a in all]
    accuracy_train = accuracy_score(all_gts, all_preds)

    loss_avg = running_loss / n_train
    if verbose: print(f"train:  {loss:>7f}  loss avg: {loss_avg:>7f}   [epoch {i_epoch:>3d}: {n_train:>5d}/{n_train:>5d}]")
    if verbose: print(f"        {loss:>7f}  accuracy: {accuracy_train:>7f}")


    # validation loop
    model.eval()
    running_loss_val = 0.
    correct = 0
    with torch.no_grad():
        for i_batch_val, (x, y) in enumerate(dl_val):
            x = x.float().to(device)
            pred_val = model(x)

            pred_class = np.argmax(pred_val.numpy(), axis=1)
            correct == (pred_class == y.numpy()).sum()

            loss_val = loss_fn(pred_val, y)
            running_loss_val += loss_val
    
    accuracy_val = correct / n_val
    loss_val_avg = running_loss_val / n_val

    if verbose: print(f"val:    loss avg: {loss_val_avg:>7f}   [epoch {i_epoch:>3d}]")
    if verbose: print(f"        accuracy: {loss_val_avg:>7f}\n")

    # write some metrics to tensorboard
    writer.add_scalars("_train_val/loss", {"train": loss_avg, "val": loss_val_avg}, i_epoch)
    writer.add_scalars("_train_val/accuracy", {"train": accuracy_train, "val": accuracy_val}, i_epoch)
    writer.add_scalar("train/loss", loss_avg, i_epoch)
    writer.add_scalar("train/accuracy", accuracy_train, i_epoch)
    writer.add_scalar("val/loss", loss_val_avg, i_epoch)
    writer.add_scalar("val/accuracy", accuracy_val, i_epoch)

    checkpoint_file = os.path.join(checkpoint_dir, f"epoch_{i_epoch}")
    if verbose: print(f"Saving current state to '{checkpoint_file}'")
    torch.save({
            'epoch': i_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            }, checkpoint_file)
    
    # TODO: write best
    # TODO: load checkpoint: https://pytorch.org/tutorials/beginner/saving_loading_models.html


