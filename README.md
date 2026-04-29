# DANet Pipeline

A PyTorch-based deep learning pipeline for tabular data classification and regression, featuring **Dual-Attention Networks (DANet)** with feature-wise self-attention and optional sample-wise attention mechanisms.

## Features

- **Dual-Attention Architecture**: Feature-level self-attention for learning complex feature interactions, plus optional sample-level attention
- **End-to-End Pipeline**: Handles preprocessing (scaling, encoding), training, evaluation, hyperparameter tuning, and model persistence
- **Hyperparameter Optimization**: Built-in Optuna integration with Bayesian optimization and early pruning
- **Production Ready**: Save/load full pipelines with preprocessing artifacts, reproducible training with seed control
- **Extensible Design**: Abstract base class makes it easy to add new task types (regression, binary/multiclass classification)

## Installation

```bash
pip install danet-pipeline