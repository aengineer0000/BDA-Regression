# Building Damage Intensity Regression

## Overview
This repository contains a machine learning pipeline for Building Damage Assessment (BDA), formulated as a continuous regression task. Using bi-temporal satellite imagery from the [xView2 dataset](https://xview2.org/), the system predicts structural damage intensity on a continuous scale from 0.0 (No Damage) to 1.0 (Destroyed). The project evaluates a range of models, from traditional tabular regressors to deep spatial Siamese networks.

## Repository Structure
* `resnet.py`: Core implementation of the Siamese ResNet-34 model, tri-modal fusion logic, and the lazy-loading data pipeline.
* `train_baseline.py`: Training script for baseline spatial models (Simple CNN, DeepDamageCNN) and tabular models (KNN, XGBoost, Random Forest).
* `random_forest.py`: Optimized Random Forest implementation featuring systematic hyperparameter tuning via `GridSearchCV`.
* `exploration.ipynb`: Data exploration and visualization notebook for analyzing class imbalances and bi-temporal image distributions.

## Installation
Ensure you have Python 3.8+ and the following dependencies installed:

```bash
pip install torch torchvision numpy pandas scikit-learn xgboost opencv-python matplotlib
```

*Note: This codebase uses XPU acceleration via the `intel_extension_for_pytorch` toolkit.*

## Usage

### 1. Data Preparation
The pipeline expects the xView2 dataset structure. Building patches are extracted centered on WKT polygons at a 224 x 224 resolution.

### 2. Training the Siamese Model
To train the primary regression model with class balancing:
```python
python resnet.py
```

### 3. Running Baselines and Hyperparameter Tuning
To perform a grid search for the Random Forest model:
```python
python random_forest.py
```
To train and view results of baseline models:
```python
python train_baseline.py
```
