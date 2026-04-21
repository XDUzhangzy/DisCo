# DisCo-iFormer

This repository provides a cleaned and modular implementation of DisCo-iFormer for EEG decoding experiments.

The codebase is organized around a simple training pipeline with separate modules for the model, dataset handling, training engine, and experiment entrypoint.

## Repository Structure

- `model.py`: core DisCo-iFormer architecture
- `dataset.py`: dataset loading, preprocessing, and fold construction
- `engine.py`: loss functions, training loop, validation, and evaluation
- `train.py`: command-line entrypoint for training experiments
- `requirements.txt`: minimal Python dependencies

## Features

- Modular model and training code
- Standard command-line training interface
- Built-in checkpoint saving and training history export
- Evaluation summary export in CSV and Excel formats
- Easy adaptation to custom EEG datasets

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The training script expects subject-level `.mat` files. Each file should contain:

- `data`: EEG trials
- `labels`: class labels
- `labels_new`: auxiliary condition labels used by the contrastive objective

## Training

Run training with:

```bash
python train.py --device cuda:0
```

You can override the default paths and hyperparameters from the command line. For example:

```bash
python train.py --data-dir /path/to/data --output-dir ./outputs/run_01 --num-epochs 100 --batch-size 128
```

## Outputs

Training outputs are saved under the configured output directory and include:

- checkpoints for the best model of each fold
- per-fold training histories in JSON format
- evaluation summaries in CSV and Excel formats

## Customization

The repository is intentionally lightweight. Common extension points are:

- adjusting dataset parsing in `dataset.py`
- modifying losses or training behavior in `engine.py`
- changing architectural details in `model.py`
- exposing new experiment settings in `train.py`

## Notes

This codebase is structured as a research implementation intended to be straightforward to inspect and modify for new experiments.
