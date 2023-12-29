# TODO: confusion matrix -> maybe in test?
# TODO: validate class balance after splits
# TODO: introduce sampler?
# TODO: change !unzip to subprocess?


import datetime
import json
import os
import time

import numpy as np

from models import TreeClassifConvNet, TreeClassifResNet50
from utils import TreeClassifPreprocessedDataset

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


################# SETTINGS ##################
######## GENERAL

# general training settings
N_epochs = 100
batch_size = 40
learning_rate = 1e-4
verbose = True

# path to checkpoint to resume training from, leave blank for training from scratch
resume = ""
dataset_dir = "data/1123_delete_nan_samples"     # just for info purposes, because in docker, data is directly in ./data
run_root = "/seminar/datscieo-0/colin/runs"

######## DATA
# create datasets and dataloaders
dataset = TreeClassifPreprocessedDataset(dataset_dir)

# split the dataset into train and validation
splits = [.7, .3]
ds_train, ds_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))

# define dataloaders for training
dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_val = DataLoader(ds_val, batch_size, shuffle=True)

if verbose: print(
    f"\nUsing dataset with properties:\n"       \
    f"\tsamples:    {len(dataset)}\n"              \
    f"\t   train:   {len(ds_train)}\n"             \
    f"\t   val:     {len(ds_val)}\n"               \
    f"\tshape: {dataset[0][0].shape}\n"         \
    )


######## MODEL
model = TreeClassifResNet50(
    n_classes = dataset.n_classes,
    width = dataset.width,
    height = dataset.height,
    depth = dataset.depth
)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if verbose:
    print(model)
    print(f"with {sum(p.numel() for p in model.parameters())} parameters")

#############################################

# device check
print("device:", device)
assert device == "cuda", "GPU not working, please check."



################# TRAINING LOOP #############

# where to save training progress info and checkpoints
run_dir = os.path.join(run_root, f"{time.strftime('%Y%m%d', time.localtime())}_{model.__class__.__name__}_lr_{learning_rate:.0e}_bs_{batch_size}")
info_dir = os.path.join(run_dir, "info")
checkpoint_dir = os.path.join(run_dir, "checkpoints")
os.makedirs(run_dir, exist_ok=True)
os.makedirs(info_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# write some info to run directory
info = {
    "dataset": dataset_dir,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "model": model.__class__.__name__,
    "optimizer": optim.__class__.__name__,
    "loss": loss_fn.__class__.__name__,
    "splits": splits,
    "indices train": ds_train.indices,
    "indices val": ds_val.indices,
}
with open(os.path.join(info_dir, "info.json"), "w") as f_info:
    json.dump(info, f_info)

writer = SummaryWriter(info_dir)

t0 = time.time()
last_info = t0
n_train = len(dl_train.dataset)
n_batch_train = np.ceil(n_train/batch_size)
its_total = n_batch_train * N_epochs
n_val = len(dl_val.dataset)
n_batch_val = np.ceil(n_val/batch_size)

# if resuming training, load state of checkpoint
if resume:
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    run_dir = checkpoint["run_dir"]
    checkpoint_dir = checkpoint["checkpoint_dir"]
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
else:
    start_epoch = 0
    best_loss = 1e10

for i_epoch in range(start_epoch, N_epochs):

    # training loop
    running_loss = 0.
    last_loss = 0.
    correct = 0
    model.train()
    for i_batch, (x, y) in enumerate(dl_train):
        assert torch.isnan(x).sum() == 0, "NaN in training data, please fix."

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        loss = loss_fn(pred, y)
        running_loss += loss.item()
        loss.backward()

        optim.step()
        optim.zero_grad()

        if verbose and (((time.time() - last_info) > 20) or (i_batch % (n_batch_train//10) == 0)):
            loss_avg = running_loss / (i_batch+1) # loss per batch
            last_info = time.time()
            loss, current = loss.item(), (i_batch + 1) * len(x)
            curr_it = i_epoch*n_batch_train + i_batch+1
            t_per_it = (time.time()-t0) / curr_it
            ETA = (its_total - curr_it) * t_per_it
            writer.add_scalar("train_intermediate/loss", loss_avg, i_epoch+current/n_train)
            print(f"train:  {loss:>7f}  loss avg: {loss_avg:>7f}   [epoch {i_epoch:>3d}: {current:>5d}/{n_train:>5d}]\t\tt/it {t_per_it:.2f}\tETA {datetime.timedelta(seconds=ETA)}\t{(datetime.datetime.now() + datetime.timedelta(seconds=ETA)).strftime('%Y-%m-%d %Hh%Mm%Ss')}")


    all = [(el, y_) for x, y in dl_train for el, y_ in zip(np.argmax(model(x.to(device)).detach().cpu().numpy(), axis=1), y.detach().cpu().numpy())]
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
    t0_val = time.time()
    with torch.no_grad():
        for i_batch_val, (x, y) in enumerate(dl_val):
            x = x.to(device)
            y = y.to(device)

            pred_val = model(x)

            pred_class = np.argmax(pred_val.cpu().numpy(), axis=1)
            correct += (pred_class == y.cpu().numpy()).sum()

            loss_val = loss_fn(pred_val, y)
            running_loss_val += loss_val
    t1_val = time.time()

    accuracy_val = correct / n_val
    loss_val_avg = running_loss_val / n_val

    if verbose: print(f"val:\t\t  loss avg: {loss_val_avg:>7f}\t\t\t\t\tt/it {(t1_val-t0_val)/n_batch_val:.2f}\tt_val {t1_val-t0_val:.2f}")
    if verbose: print(f"    \t\t  accuracy: {accuracy_val:>7f}")

    # write some metrics to tensorboard
    writer.add_scalars("_train_val/loss", {"train": loss_avg, "val": loss_val_avg}, i_epoch+1)
    writer.add_scalars("_train_val/accuracy", {"train": accuracy_train, "val": accuracy_val}, i_epoch+1)
    writer.add_scalar("train/loss", loss_avg, i_epoch+1)
    writer.add_scalar("train/accuracy", accuracy_train, i_epoch+1)
    writer.add_scalar("val/loss", loss_val_avg, i_epoch+1)
    writer.add_scalar("val/accuracy", accuracy_val, i_epoch+1)

    t_per_epoch = (time.time()-t0)/(i_epoch+1)
    if verbose: print(f"t/epoch {t_per_epoch:.2f} s")

    checkpoint_file = os.path.join(checkpoint_dir, f"epoch_{i_epoch}.pth")
    if verbose: print(f"Saving current state to '{checkpoint_file}'\n")
    torch.save({
            "epoch": i_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "train_loss": loss_avg,
            "val_loss": loss_val_avg,
            "best_loss": best_loss,
            "run_dir": run_dir,
            "checkpoint_dir": checkpoint_dir
            }, checkpoint_file)
    
    if loss_val_avg < best_loss:
        checkpoint_file = os.path.join(info_dir, f"best.pth")
        if verbose: print(f"Saving current best state to '{checkpoint_file}'\n")
        torch.save({
                "epoch": i_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "train_loss": loss_avg,
                "val_loss": loss_val_avg,
                "best_loss": best_loss,
                "run_dir": run_dir,
                "checkpoint_dir": checkpoint_dir
                }, checkpoint_file)

        best_loss = loss_val_avg
