import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset import compute_class_weights, create_intra_subject_splits, list_subject_files, load_subject_data, set_seed
from engine import evaluate_model, train_one_fold
from model import DisCoIFormer


def build_parser():
    parser = argparse.ArgumentParser(description="Train DisCo-iFormer on RSVP_Tsinghua_new.")
    parser.add_argument("--data-dir", type=str, default="/media/zhangzy/Files/Code/dataset/Research2Dataset/RSVP_Tsinghua_new")
    parser.add_argument("--output-dir", type=str, default="./outputs/disco_iformer_tsinghua")
    parser.add_argument("--condition-range", type=int, default=3)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--n-timepoints", type=int, default=250)
    parser.add_argument("--n-channels", type=int, default=62)
    parser.add_argument("--inner-channels", type=int, default=32)
    parser.add_argument("--n-class", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-cls-token", action="store_true")
    return parser


def prepare_output_dirs(output_dir):
    checkpoints_dir = output_dir / "checkpoints"
    histories_dir = output_dir / "histories"
    results_dir = output_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    histories_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir, histories_dir, results_dir


def resolve_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    return torch.device(device_name if device_name.startswith("cuda") else "cpu")


def main(args):
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    checkpoints_dir, histories_dir, results_dir = prepare_output_dirs(output_dir)
    subject_files = list_subject_files(args.data_dir)
    all_fold_metrics = []

    for subject_file in subject_files:
        subject_name = subject_file.stem
        print(f"\n=== {subject_name} ===")
        eeg, labels, condition_labels = load_subject_data(subject_file, keep_range=args.condition_range)
        class_weights = compute_class_weights(labels)

        for fold_index in range(args.num_folds):
            print(f"[{subject_name}] fold {fold_index + 1}/{args.num_folds}")
            train_dataset, val_dataset, test_dataset = create_intra_subject_splits(
                eeg=eeg,
                labels=labels,
                condition_labels=condition_labels,
                fold_index=fold_index,
                num_folds=args.num_folds,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )

            model = DisCoIFormer(args).to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay,
            )

            best_checkpoint, history = train_one_fold(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                device=device,
                num_epochs=args.num_epochs,
            )

            if best_checkpoint is None:
                raise RuntimeError(f"No valid training step completed for {subject_name} fold {fold_index + 1}.")

            model.load_state_dict(best_checkpoint["model_state_dict"])
            metrics = evaluate_model(model, test_loader, device)
            metrics["subject"] = subject_name
            metrics["fold"] = fold_index + 1
            metrics["best_epoch"] = best_checkpoint["best_epoch"]
            all_fold_metrics.append(metrics)

            torch.save(best_checkpoint, checkpoints_dir / f"{subject_name}_fold{fold_index + 1}.pt")
            (histories_dir / f"{subject_name}_fold{fold_index + 1}.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            print(
                f"AUC={metrics['auc']:.4f} "
                f"F1={metrics['f1']:.4f} "
                f"BalancedAcc={metrics['balanced_accuracy']:.4f} "
                f"Precision={metrics['precision']:.4f}"
            )

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    summary_df = fold_metrics_df.drop(columns=["subject", "fold"]).agg(["mean", "std"]).transpose().reset_index()
    summary_df = summary_df.rename(columns={"index": "metric"})

    fold_metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)

    with pd.ExcelWriter(results_dir / "metrics.xlsx") as writer:
        fold_metrics_df.to_excel(writer, sheet_name="fold_metrics", index=False)
        summary_df.to_excel(writer, sheet_name="summary_metrics", index=False)

    print("\nFinished. Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
