# TODO: validate class balance after splits
# TODO: introduce sampler?
# TODO: change !unzip to subprocess?
# TODO: visualize tensorboard on server?


import datetime
import json
import os, sys
import time

from argparse import ArgumentParser, Namespace

import numpy as np

import models
from utils import TreeClassifPreprocessedDataset
from utils import TorchStandardScaler

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils import train_schedules


parser = ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100)
parser.add_argument("-b", "--batch_size", type=int, default=40)
parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)

parser.add_argument("-d", "--dataset_dir", type=str, default="/seminar/datscieo-0/data/train_val/train_val_delete_nan_bands")
parser.add_argument("-i", "--indices", type=str, default="")
parser.add_argument("--ignore-augmentations", nargs="+", default=[])

parser.add_argument("-s", "--scaler", type=str, default="")

parser.add_argument("-m", "--model", type=str, default="TreeClassifResNet50")
parser.add_argument("-r", "--resume", type=str, default="")

parser.add_argument("-R", "--run_root", type=str, default="/seminar/datscieo-0/runs")
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--schedule", default="", help="#TODO")

parser.add_argument("-v", "--verbose", type=bool, default=True)


initial_args = parser.parse_args()
if initial_args.schedule:
    schedule = getattr(train_schedules, initial_args.schedule)
else:
    schedule = ["single train"]



for train_params in schedule:
    if not train_params == "single train":
        args = parser.parse_args(namespace=Namespace(**train_params))
    else:
        args = initial_args
    ################# SETTINGS ##################
    ######## GENERAL

    # general training settings
    N_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    verbose = args.verbose

    # path to checkpoint to resume training from, leave blank for training from scratch
    resume = args.resume
    dataset_dir = args.dataset_dir
    
    if initial_args.schedule:
        run_root = os.path.join(args.run_root, initial_args.schedule)
    else:
        run_root = args.run_root

    ######## DATA
    # create datasets and dataloaders
    dataset = TreeClassifPreprocessedDataset(dataset_dir, indices=eval(args.indices) if args.indices else None)

    # split the dataset into train and validation
    splits = [.8, .2]
    ds_train, ds_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))

    # define dataloaders for training
    dl_train = DataLoader(ds_train, batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size, shuffle=True)

    # define an optional scaler
    scaler = getattr(sys.modules[__name__], args.scaler)() if args.scaler else None

    dataset_info = f"\nUsing dataset with properties:\n"    \
        f"\tsamples:    {len(dataset)}\n"                   \
        f"\t   train:   {len(ds_train)}\n"                  \
        f"\t   val:     {len(ds_val)}\n"                    \
        f"\tshape: {dataset[0][0].shape}\n"                 \
        f"\split: {splits}\n"                               \
        f"\tScaler: {scaler}"
    if verbose: print(dataset_info)


    ######## MODEL
    model = getattr(models, args.model)(
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
    # assert device == "cuda", "GPU not working, please check."



    ################# TRAINING LOOP #############

    # where to save training progress info and checkpoints
    run_name = args.run_name or f"{model.__class__.__name__}_lr_{learning_rate:.0e}_bs_{batch_size}"
    run_dir = os.path.join(run_root, run_name)
    info_dir = os.path.join(run_dir, "info")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # fit scaler
    if scaler:
        # t0_scale = time.time()
        X = torch.tensor(np.array([x[0] for x in dataset]))
        print("X shape for fitting", X.shape)
        scaler.fit(X)
        # scaler.save(os.path.join(run_root, "scaler.pth"))
        # t1_scale = time.time()
        # print(f"Scaling time: {t1_scale-t0_scale:.3f} s")

    # write some info to run directory
    info = {
        "dataset": dataset_dir,
        "dataset_info": dataset_info,
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

            if scaler: x = scaler.transform(x)
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
                if scaler: x = scaler.transform(x)

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
                "best_accuracy": accuracy_val,
                "run_dir": run_dir,
                "checkpoint_dir": checkpoint_dir,
                "time": time.strftime('%Y%m%d', time.localtime())
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
                    "best_accuracy": accuracy_val,
                    "run_dir": run_dir,
                    "checkpoint_dir": checkpoint_dir,
                    "time": time.strftime('%Y%m%d', time.localtime())
                    }, checkpoint_file)

            best_loss = loss_val_avg
