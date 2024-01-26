import os

from utils.focalloss import FocalLoss

data_root = "data" if os.name == "nt" else "/seminar/datscieo-0/data"
run_root = "/seminar/datscieo-0/runs_baseline"
epochs = 30

# baselines
ignore_augment = "all"
train_baseline_delete_nan = {
    "epochs": 250,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "baseline_delete_nan_samples_long",

    "verbose": True
}
train_baseline_no_autumn = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples_B11_2-B12_2-B2_2-B3_2-B4_2-B5_2-B6_2-B7_2-B8A_2-B8_2"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "baseline_remove_autumn",

    "verbose": True
}
train_baseline_replace_nan = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_replace_nan_fill_nan_-1"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "baseline_replace_nan",

    "verbose": True
}
train_baseline_nan_mask = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_apply_nan_mask_fill_nan_-1"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "baseline_apply_nan_mask",

    "verbose": True
}



# full augmentation baselines
ignore_augment = ["adjust_gamma"]
prefix = "full_augmentation_"
full_augmentation_baseline_delete_nan = {
    "epochs": epochs,
    "batch_size": 250,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": prefix + "delete_nan_samples_long",

    "verbose": True
}
full_augmentation_baseline_no_autumn = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples_B11_2-B12_2-B2_2-B3_2-B4_2-B5_2-B6_2-B7_2-B8A_2-B8_2"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": prefix + "remove_autumn",

    "verbose": True
}
full_augmentation_baseline_replace_nan = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_replace_nan_fill_nan_-1"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": prefix + "replace_nan",

    "verbose": True
}
full_augmentation_baseline_nan_mask = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_apply_nan_mask_fill_nan_-1"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": prefix + "apply_nan_mask",

    "verbose": True
}


# no scale
train_no_scale = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_apply_nan_mask_fill_nan_-1"),
    "indices": "",
    "ignore-augmentations": "all",
    "scaler": "",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "simple",

    "verbose": True
}
train_no_scale_w_augm = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_apply_nan_mask_fill_nan_-1"),
    "indices": "",
    "ignore-augmentations": [],
    "scaler": "",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "simple_augm",

    "verbose": True
}


# pretrained
resnet18 = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ["adjust_gamma"],
    "scaler": "TorchStandardScaler",

    "model": "resnet18",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "resnet18",

    "verbose": True
}
resnet50 = {
    "epochs": epochs,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ["adjust_gamma"],
    "scaler": "TorchStandardScaler",

    "model": "resnet50",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "resnet50",

    "verbose": True
}


# weighted
baseline_weighted = {
    "epochs": 50,
    "batch_size": 40,
    "learning_rate": 1e-4,
    "weighted_loss": True,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "weighted_baseline",

    "verbose": True
}
resnet_dropout_weighted = {
    "epochs": 50,
    "batch_size": 40,
    "learning_rate": 1e-4,
    "weighted_loss": True,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifResNet50",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "weighted_resnet_dropout",

    "verbose": True
}

# weighted
baseline_weighted2 = {
    "epochs": 50,
    "batch_size": 40,
    "learning_rate": 1e-1,
    "weighted_loss": True,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "weighted_baseline2",

    "verbose": True
}
resnet_dropout_weighted2 = {
    "epochs": 50,
    "batch_size": 40,
    "learning_rate": 1e-1,
    "weighted_loss": True,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifResNet50Dropout",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "weighted_resnet_dropout2",

    "verbose": True
}

# focal
baseline_focal = {
    "epochs": 50,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "loss": "FocalLoss",
    "weighted_loss": False,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "focal_baseline",

    "verbose": True
}
resnet_dropout_focal = {
    "epochs": 50,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "loss": "FocalLoss",
    "weighted_loss": False,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifResNet50",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "focal_resnet_dropout",

    "verbose": True
}

resnet50_dropout75 = {
    "epochs": 100,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifResNet50Dropout75",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "TreeClassifResNet50Dropout75",

    "verbose": True
}

# testing events
mini1 = {
    "epochs": 1,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_delete_nan_samples"),
    "indices": "range(100)",
    "ignore-augmentations": "all",
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "baseline_delete_nan_samples",

    "verbose": True
}
mini2 = {
    "epochs": 1,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": os.path.join(data_root, "train_val/train_val_replace_nan_fill_nan_-1_B11_2-B12_2-B2_2-B3_2-B4_2-B5_2-B6_2-B7_2-B8A_2-B8_2"),
    "indices": "range(200)",
    "ignore-augmentations": "all",
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else run_root,
    "run_name": "baseline_delete_nan_samples",

    "verbose": True
}



schedule_baseline = [
    train_baseline_delete_nan,
    train_baseline_no_autumn,
    full_augmentation_baseline_replace_nan,
    full_augmentation_baseline_no_autumn
]


schedule_no_scale = [
    train_no_scale,
    train_no_scale_w_augm
]

schedule_full_augmentation = [
    full_augmentation_baseline_delete_nan,
    full_augmentation_baseline_no_autumn
    ]


schedule_mini = [
    mini1,
    mini2
]





schedule_baseline_old = [
    train_baseline_delete_nan,
    train_baseline_no_autumn,
    # train_baseline_replace_nan,   # error, NaN in data!
    # train_baseline_nan_mask
    ]