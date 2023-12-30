from argparse import ArgumentParser
import datetime
import json
import os
import time

import numpy as np

import models
from utils import TreeClassifPreprocessedDataset

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader, random_split


parser = ArgumentParser()
parser.add_argument("run_id", type=str)
group = parser.add_mutually_exclusive_group()
group.add_argument("epoch", type=int)
group.add_argument("checkpoint", type=str)
parser.add_argument("-R", "--run_root", type=str, default="/seminar/datscieo-0/colin/runs")
parser.add_argument("-d", "--dataset_dir", type=str, default="data/1123_delete_nan_samples")
parser.add_argument("-i", "--indices", type=str, default="")
parser.add_argument("-m", "--model", type=str, default="TreeClassifResNet50")

parser.add_argument("-b", "--batch_size", type=int, default=40)
parser.add_argument("-v", "--verbose", type=bool, default=True)

args = parser.parse_args()


################# SETTINGS ##################
######## GENERAL
run_id = "20231229_TreeClassifResNet50"
run_id = "20231220-20h53m17s_97epochs"
run_id = args.run_id
checkpoint_epoch = args.epoch
batch_size_test = args.batch_size
verbose = True

######## DATA
# create datasets and dataloaders
dataset_dir = args.dataset_dir
dataset = TreeClassifPreprocessedDataset(dataset_dir, indices=eval(args.indices) if args.indices else None)

# at this point, perform prediction on validation split, later use real test set
splits = [.7, .3]
ds_train, ds_val = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
ds_test = ds_val.dataset
dl_test = DataLoader(ds_test, batch_size_test, shuffle=True)

if verbose: print(
    f"\nUsing dataset with properties:\n"       \
    f"\tsamples test:    {len(ds_test)}\n"      \
    f"\tshape: {ds_test[0][0].shape}\n"         \
    )


######## MODEL
# model = TreeClassifResNet50(
model = getattr(models, args.model)(
    n_classes = dataset.n_classes,
    width = dataset.width,
    height = dataset.height,
    depth = dataset.depth
)

if verbose:
    print(model)
    print(f"with {sum(p.numel() for p in model.parameters())} parameters")
#############################################


checkpoint = torch.load(os.path.join(args.run_root, run_id, "checkpoints", f"epoch_{checkpoint_epoch}.pth"))
output_dir = os.path.join(args.run_root, run_id, "eval")
os.makedirs(output_dir, exist_ok=True)

model.load_state_dict(checkpoint["model_state_dict"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

n_batch_test = np.ceil(len(ds_test)/batch_size_test)
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
        print(f"train:  {correct:>4d}     [{current:>5d}/{len(ds_test):>5d}]\t\tt/it {t_per_it:.2f}\tETA {datetime.timedelta(seconds=ETA)}\t{(datetime.datetime.now() + datetime.timedelta(seconds=ETA)).strftime('%Y-%m-%d %Hh%Mm%Ss')}")


################# EVALUATION ################
accuracy_test = accuracy_score(all_gts, all_preds)
if verbose: print("accuracy:", accuracy_test)


cm = confusion_matrix(all_gts, all_preds, labels=range(dataset.n_classes))
if verbose: print("confusion matrix\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ds_test.classes)
disp.plot(colorbar=True, cmap="Blues")
tick_size = 8
plt.xticks(rotation=30, ha='right', size=tick_size)
plt.yticks(size=tick_size)
label_size = 18
plt.xlabel("Predicted label", size=label_size)
plt.ylabel("True label", size=label_size)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.98)


# save results
with open(os.path.join(output_dir, f"metrics_{checkpoint_epoch}.txt"), "w") as f:
    json.dump({
        "accuracy": accuracy_test,
        "confusion_matrix": cm.tolist()
    }, f)
plt.savefig(os.path.join(output_dir, f"confusion_{checkpoint_epoch}.png"),dpi=300, bbox_inches="tight")