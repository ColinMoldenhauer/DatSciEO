import os

# baselines
ignore_augment = "all"
train_baseline_nan_samples = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_delete_nan_samples",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "baseline_delete_nan_samples",

    "verbose": True
}
train_baseline_no_autumn = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_delete_nan_samples_B11_2-B12_2-B2_2-B3_2-B4_2-B5_2-B6_2-B7_2-B8A_2-B8_2",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "baseline_remove_autumn",

    "verbose": True
}
train_baseline_replace_nan = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_replace_nan_fill_nan_-1",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "baseline_replace_nan",

    "verbose": True
}
train_baseline_nan_mask = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_apply_nan_mask_fill_nan_-1",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "baseline_apply_nan_mask",

    "verbose": True
}


# full augmentation baselines
ignore_augment = ["adjust_gamma"]
prefix = "full_augmentation_"
full_augmentation_baseline_nan_samples = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_delete_nan_samples",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": prefix + "delete_nan_samples",

    "verbose": True
}
full_augmentation_baseline_no_autumn = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_delete_nan_samples_B11_2-B12_2-B2_2-B3_2-B4_2-B5_2-B6_2-B7_2-B8A_2-B8_2",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": prefix + "remove_autumn",

    "verbose": True
}
full_augmentation_baseline_replace_nan = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_replace_nan_fill_nan_-1",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": prefix + "replace_nan",

    "verbose": True
}
full_augmentation_baseline_nan_mask = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_apply_nan_mask_fill_nan_-1",
    "indices": "",
    "ignore-augmentations": ignore_augment,
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": prefix + "apply_nan_mask",

    "verbose": True
}


# other
train_no_scale = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_apply_nan_mask_fill_nan_-1",
    "indices": "",
    "ignore-augmentations": "all",
    "scaler": "",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "simple",

    "verbose": True
}
train_no_scale_w_augm = {
    "epochs": 20,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_apply_nan_mask_fill_nan_-1",
    "indices": "",
    "ignore-augmentations": [],
    "scaler": "",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "simple_augm",

    "verbose": True
}



# testing events
mini1 = {
    "epochs": 1,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_delete_nan_samples",
    "indices": "range(100)",
    "ignore-augmentations": "all",
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "baseline_delete_nan_samples",

    "verbose": True
}
mini2 = {
    "epochs": 1,
    "batch_size": 40,
    "learning_rate": 1e-4,

    "dataset_dir": "runs/data/train_val" if os.name == "nt" else "/seminar/datscieo-0/data/train_val/train_val_replace_nan_fill_nan_-1_B11_2-B12_2-B2_2-B3_2-B4_2-B5_2-B6_2-B7_2-B8A_2-B8_2",
    "indices": "range(200)",
    "ignore-augmentations": "all",
    "scaler": "TorchStandardScaler",

    "model": "TreeClassifConvNet",
    "resume": "",

    "run_root": "runs" if os.name == "nt" else "/seminar/datscieo-0/runs",
    "run_name": "baseline_delete_nan_samples",

    "verbose": True
}


schedule_no_scale = [
    train_no_scale,
    train_no_scale_w_augm
]



schedule_baseline = [
    # train_baseline_nan_samples,
    # train_baseline_no_autumn,
    # train_baseline_replace_nan,   # error, NaN in data!
    train_baseline_nan_mask
    ]


schedule_full_augmentation = [
    full_augmentation_baseline_nan_samples,
    full_augmentation_baseline_no_autumn,
    full_augmentation_baseline_replace_nan,
    full_augmentation_baseline_nan_mask
    ]


schedule_mini = [
    mini1,
    mini2
]