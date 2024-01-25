from argparse import ArgumentParser
import datetime
import json
import os
import time

import numpy as np

import models
from utils import TreeClassifPreprocessedDataset, confusion_matrix_and_classf_metrics

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader, random_split


data_root = "data" if os.name == "nt" else "/seminar/datscieo-0/data"


def test_model(model, device, dl_test, batch_size_test):
    n_samples = len(dl_test.dataset)
    n_batch_test = np.ceil(n_samples/batch_size_test)
    correct = 0
    last_info = time.time()
    t0 = time.time()
    model.eval()
    all_gts = []
    all_preds = []
    ################# PREDICTION LOOP ###########
    for i_batch, (x, y) in enumerate(dl_test):
        assert torch.isnan(x).sum() == 0, "NaN in test data, please fix."

        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        all_gts.extend(y.detach().cpu().numpy())
        all_preds.extend(np.argmax(pred.detach().cpu().numpy(), axis=1))


        if verbose and (((time.time() - last_info) > 20) or (i_batch % (n_batch_test//10) == 0)):
            last_info = time.time()
            correct, current = (np.argmax(pred.detach().cpu().numpy(), axis=1) == y.detach().cpu().numpy()).sum(), (i_batch + 1) * len(x)
            t_per_it = (time.time()-t0) / (i_batch+1)
            ETA = (n_batch_test - i_batch - 1) * t_per_it
            print(f"train:  {correct:>4d}     [{current:>5d}/{n_samples:>5d}]\t\tt/it {t_per_it:.2f}\tETA {datetime.timedelta(seconds=ETA)}\t{(datetime.datetime.now() + datetime.timedelta(seconds=ETA)).strftime('%Y-%m-%d %Hh%Mm%Ss')}")

    return all_gts, all_preds

parser = ArgumentParser()
parser.add_argument("run_id", type=str)
parser.add_argument("--suffix", type=str, default="test")
parser.add_argument("--do_split", action="store_true", default=False)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--epoch", type=int)
group.add_argument("--checkpoint", type=str)
parser.add_argument("-R", "--run_root", type=str, default="runs" if os.name == "nt" else "/seminar/datscieo-0/runs_baseline")
parser.add_argument("-d", "--dataset_dir", type=str, default=os.path.join(data_root, "test/test_delete_nan_samples"))
parser.add_argument("-i", "--indices", type=str, default="")
parser.add_argument("-m", "--model", type=str, default="TreeClassifResNet50")

parser.add_argument("-b", "--batch_size", type=int, default=40)
parser.add_argument("-v", "--verbose", type=bool, default=True)

args = parser.parse_args()

################# SETTINGS ##################
######## GENERAL
run_id = args.run_id
batch_size_test = args.batch_size
verbose = args.verbose

######## DATA
# create datasets and dataloaders
dataset_dir = args.dataset_dir

# optionally split for testing on validation set
dataset = TreeClassifPreprocessedDataset(dataset_dir, indices=eval(args.indices) if args.indices else None)
if args.do_split:
    splits = [.8, .2]
    _, ds_test = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    ds_test = ds_test.dataset
else:
    ds_test = dataset
dl_test = DataLoader(ds_test, batch_size_test, shuffle=True)

if verbose: print(
    f"\nUsing dataset with properties:\n"       \
    f"\tsamples test:    {len(ds_test)}\n"      \
    f"\tshape: {ds_test[0][0].shape}\n"         \
    )


######## MODEL
model = getattr(models, args.model)(
    n_classes = ds_test.n_classes,
    width = ds_test.width,
    height = ds_test.height,
    depth = ds_test.depth
)

if verbose:
    print(model)
    print(f"with {sum(p.numel() for p in model.parameters())} parameters")
#############################################

map_loc = torch.device('cpu') if os.name == "nt" else None
if args.checkpoint == "best":
    checkpoint = torch.load(os.path.join(args.run_root, run_id, f"best.pth"), map_location=map_loc)
elif args.checkpoint:
    checkpoint = torch.load(os.path.join(args.checkpoint), map_location=map_loc)
else:
    checkpoint = torch.load(os.path.join(args.run_root, run_id, "checkpoints", f"epoch_{args.epoch}.pth"), map_location=map_loc)
output_dir = os.path.join(args.run_root, run_id, "eval")
os.makedirs(output_dir, exist_ok=True)

model.load_state_dict(checkpoint["model_state_dict"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

all_gts, all_preds = test_model(model, device, dl_test, batch_size_test)


################# EVALUATION ################
suffix = args.suffix if args.suffix.startswith("_") else "_" + args.suffix
np.save(os.path.join(output_dir, f"all_preds{suffix}.npy"), all_preds)
np.save(os.path.join(output_dir, f"all_gts{suffix}.npy"), all_gts)
confusion_matrix_and_classf_metrics(all_gts, all_preds, ds_test, output_dir, verbose=verbose, titleConfMatrix="", filename=f"CM_{args.run_id}", suffix=suffix)