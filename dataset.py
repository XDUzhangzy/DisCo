"""
Dataset loading and fold construction utilities for RSVP_Tsinghua_new.
"""

import os
import random
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import TensorDataset


def set_seed(seed=1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def standardize_eeg(data, channels=False, eps=1e-6):
    if channels:
        mean = data.mean(dim=-1, keepdim=True)
        std = data.std(dim=-1, keepdim=True)
    else:
        mean = data.mean(dim=(-1, -2), keepdim=True)
        std = data.std(dim=(-1, -2), keepdim=True)
    return (data - mean) / (std + eps)


def extract_subject_id(path):
    match = re.search(r"sub_(\d+)\.mat$", path.name)
    if not match:
        raise ValueError(f"Unexpected subject filename: {path.name}")
    return int(match.group(1))


def list_subject_files(data_dir):
    data_dir = Path(data_dir)
    subject_files = sorted(data_dir.glob("sub_*.mat"), key=extract_subject_id)
    if not subject_files:
        raise FileNotFoundError(f"No subject files found under {data_dir}")
    return subject_files


def remap_condition_labels(condition_labels, keep_range=3):
    condition_labels = condition_labels.clone()
    if keep_range == 6:
        return condition_labels
    high_start = 7 + keep_range
    low_start = 1 + keep_range
    high_mask = (condition_labels >= high_start) & (condition_labels < 13)
    low_mask = (condition_labels >= low_start) & (condition_labels <= 6)
    condition_labels[high_mask | low_mask] = 0
    return condition_labels


def load_subject_data(subject_file, keep_range=3):
    mat = sio.loadmat(str(subject_file))
    eeg = torch.tensor(mat["data"], dtype=torch.float32)
    labels = torch.tensor(mat["labels"], dtype=torch.long).permute(1, 0)
    condition_labels = torch.tensor(mat["labels_new"], dtype=torch.long).permute(1, 0)
    condition_labels = remap_condition_labels(condition_labels, keep_range=keep_range)
    eeg = eeg.unsqueeze(0).permute(3, 0, 1, 2)
    eeg = standardize_eeg(eeg, channels=True)
    return eeg, labels, condition_labels


def compute_class_weights(labels):
    labels = labels.view(-1)
    num_positive = int(labels.sum().item())
    num_negative = int(labels.numel() - num_positive)
    if num_positive == 0 or num_negative == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    return torch.tensor([1.0, num_negative / num_positive], dtype=torch.float32)


def build_fold_indices(num_examples, num_folds=3):
    return [torch.tensor(block, dtype=torch.long) for block in np.array_split(np.arange(num_examples), num_folds)]


def create_intra_subject_splits(eeg, labels, condition_labels, fold_index, num_folds=3):
    if num_folds < 3:
        raise ValueError("num_folds must be at least 3.")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index must be in [0, {num_folds - 1}].")

    fold_indices = build_fold_indices(len(eeg), num_folds)
    test_fold = fold_index
    val_fold = (fold_index + 1) % num_folds
    train_folds = [i for i in range(num_folds) if i not in (test_fold, val_fold)]

    train_indices = torch.cat([fold_indices[i] for i in train_folds], dim=0)
    val_indices = fold_indices[val_fold]
    test_indices = fold_indices[test_fold]

    train_dataset = TensorDataset(eeg[train_indices], labels[train_indices], condition_labels[train_indices])
    val_dataset = TensorDataset(eeg[val_indices], labels[val_indices], condition_labels[val_indices])
    test_dataset = TensorDataset(eeg[test_indices], labels[test_indices], condition_labels[test_indices])
    return train_dataset, val_dataset, test_dataset
