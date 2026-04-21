"""
Training, validation, and evaluation utilities.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import Optimizer
from tqdm import tqdm


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_losses=2):
        super().__init__()
        self.num_losses = num_losses
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)

    def forward(self, *losses):
        if len(losses) != 2 or self.num_losses != 2:
            raise ValueError("UncertaintyWeightedLoss currently supports exactly two losses.")
        loss_a, loss_b = losses
        lambda_a = 1 / self.weights[0] ** 2
        lambda_b = 1 / self.weights[1] ** 2
        return lambda_a * loss_a + lambda_b * loss_b + torch.log(self.weights[0] * self.weights[1])


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        labels = labels.squeeze(-1)
        features = F.normalize(features, dim=1)
        similarity = torch.matmul(features, features.t())
        logits = torch.exp(similarity / self.temperature)

        diagonal_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal_mask, 0.0)

        positive_mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)) & ~diagonal_mask
        negative_mask = ~labels.unsqueeze(1).eq(labels.unsqueeze(0))

        positive_scores = (logits * positive_mask).sum(dim=1)
        negative_scores = (logits * negative_mask).sum(dim=1)
        valid_samples = positive_mask.sum(dim=1) > 0

        if not valid_samples.any():
            return features.new_tensor(0.0)

        positive_scores = positive_scores[valid_samples]
        negative_scores = negative_scores[valid_samples]
        return (-torch.log(positive_scores / (positive_scores + negative_scores + 1e-12) + 1e-12)).mean()


class ValidationLRScheduler:
    def __init__(self, optimizer: Optimizer, patience=5, min_lr=1e-6, factor=0.5):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch Optimizer.")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            verbose=True,
        )

    def step(self, validation_loss):
        self.scheduler.step(validation_loss)


class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def step(self, validation_loss):
        if self.best_loss is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def run_epoch(model, dataloader, optimizer, classification_loss, contrastive_loss, multitask_loss, device, training=True):
    model.train(mode=training)
    total_loss = 0.0
    total_classification_loss = 0.0
    total_contrastive_loss = 0.0
    num_steps = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for eeg, labels, condition_labels in tqdm(dataloader, leave=False):
            eeg = eeg.to(device, non_blocking=True)
            labels = labels.squeeze(-1).to(device, non_blocking=True)
            condition_labels = condition_labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, features = model(eeg)
            cls_loss = classification_loss(logits, labels)
            ctr_loss = contrastive_loss(features, condition_labels)
            loss = multitask_loss(cls_loss, ctr_loss)

            if not torch.isfinite(loss):
                continue

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_classification_loss += cls_loss.item()
            total_contrastive_loss += ctr_loss.item()
            num_steps += 1

    if num_steps == 0:
        return {
            "loss": float("inf"),
            "classification_loss": float("inf"),
            "contrastive_loss": float("inf"),
        }

    return {
        "loss": total_loss / num_steps,
        "classification_loss": total_classification_loss / num_steps,
        "contrastive_loss": total_contrastive_loss / num_steps,
    }


def collect_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for eeg, labels, _condition_labels in tqdm(dataloader, leave=False):
            eeg = eeg.to(device, non_blocking=True)
            logits, _features = model(eeg)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(probabilities.detach().cpu().numpy())
            all_predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.extend(labels.squeeze(-1).cpu().numpy())

    return np.asarray(all_labels), np.asarray(all_predictions), np.asarray(all_scores)


def compute_classification_metrics(labels, predictions, scores):
    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()

    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": accuracy_score(labels, predictions),
        "auc": auc,
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "true_positive_rate": tp / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
    }


def evaluate_model(model, dataloader, device):
    labels, predictions, scores = collect_predictions(model, dataloader, device)
    return compute_classification_metrics(labels, predictions, scores)


def train_one_fold(model, optimizer, train_loader, val_loader, class_weights, device, num_epochs):
    classification_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
    contrastive_loss = SupervisedContrastiveLoss().to(device)
    multitask_loss = UncertaintyWeightedLoss(num_losses=2).to(device)
    scheduler = ValidationLRScheduler(optimizer)
    early_stopping = EarlyStopping()

    best_validation_loss = float("inf")
    best_checkpoint = None
    history = []

    for epoch in range(1, num_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            classification_loss,
            contrastive_loss,
            multitask_loss,
            device,
            training=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            classification_loss,
            contrastive_loss,
            multitask_loss,
            device,
            training=False,
        )

        scheduler.step(val_metrics["loss"])
        early_stopping.step(val_metrics["loss"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_classification_loss": train_metrics["classification_loss"],
                "train_contrastive_loss": train_metrics["contrastive_loss"],
                "val_loss": val_metrics["loss"],
                "val_classification_loss": val_metrics["classification_loss"],
                "val_contrastive_loss": val_metrics["contrastive_loss"],
            }
        )

        if val_metrics["loss"] < best_validation_loss:
            best_validation_loss = val_metrics["loss"]
            best_checkpoint = {
                "model_state_dict": deepcopy(model.state_dict()),
                "loss_state_dict": deepcopy(multitask_loss.state_dict()),
                "best_epoch": epoch,
            }

        if early_stopping.should_stop:
            break

    return best_checkpoint, history
