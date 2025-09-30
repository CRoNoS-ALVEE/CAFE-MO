# CAFE-MO: Green Computing Multi-Objective Optimization

This framework supports fairness-aware, energy-efficient machine learning model optimization across multiple popular algorithms.

## Available Models

The following machine learning models are now supported:

### 1. **Random Forest** (`run_rf.py`)
- **Hyperparameters**: n_estimators, max_depth, min_samples_split
- **Use case**: Robust ensemble method, good baseline performance
- **Command**: `python scripts\run_rf.py --dataset ADULT --rounds 20`

### 2. **Logistic Regression** (`run_logreg.py`)
- **Hyperparameters**: C (regularization), max_iter
- **Use case**: Linear model, interpretable, fast training
- **Command**: `python scripts\run_logreg.py --dataset ADULT --rounds 20`

### 3. **Support Vector Machine** (`run_svm.py`)
- **Hyperparameters**: C (regularization), gamma
- **Use case**: Effective for high-dimensional data
- **Command**: `python scripts\run_svm.py --dataset ADULT --rounds 20`

### 4. **XGBoost** (`run_xgb.py`) ⭐ **NEW**
- **Hyperparameters**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda
- **Use case**: State-of-the-art gradient boosting, excellent performance
- **Command**: `python scripts\run_xgb.py --dataset ADULT --rounds 20`

### 5. **CatBoost** (`run_catboost.py`) ⭐ **NEW**
- **Hyperparameters**: iterations, depth, learning_rate, l2_leaf_reg, subsample, colsample_bylevel
- **Use case**: Handles categorical features well, robust to overfitting
- **Command**: `python scripts\run_catboost.py --dataset ADULT --rounds 20`

### 6. **LightGBM** (`run_lgbm.py`) ⭐ **NEW**
- **Hyperparameters**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, num_leaves, min_child_samples
- **Use case**: Fast training, memory efficient, excellent performance
- **Command**: `python scripts\run_lgbm.py --dataset ADULT --rounds 20`

### 7. **Gradient Boosting** (`run_gb.py`) ⭐ **NEW**
- **Hyperparameters**: n_estimators, max_depth, learning_rate, subsample, max_features, min_samples_split, min_samples_leaf
- **Use case**: Sklearn's gradient boosting, good baseline for ensemble methods
- **Command**: `python scripts\run_gb.py --dataset ADULT --rounds 20`

## Supported Datasets

- **ADULT**: Income prediction (>50K vs ≤50K)
- **COMPAS**: Recidivism prediction
- **GERMANCREDIT**: Credit risk assessment
- **LAWSCHOOL**: Bar exam pass prediction

## Multi-Model Comparison

Use the comprehensive comparison script to run multiple models:

```bash
# Run all models
python scripts\run_all_models.py --dataset ADULT --rounds 10

# Run specific models
python scripts\run_all_models.py --dataset ADULT --rounds 10 --models XGB CATBOOST LGBM

# Run with custom parameters
python scripts\run_all_models.py --dataset COMPAS --rounds 15 --tau 0.15 --alpha 0.05
```

## Quick Performance Comparison

Based on initial tests with ADULT dataset (3 rounds):

| Model | Best Accuracy | Energy Efficiency | Training Speed | Fairness |
|-------|--------------|-------------------|----------------|----------|
| XGBoost | ~86% | Moderate | Fast | Good |
| CatBoost | ~86% | Moderate | Medium | Good |
| LightGBM | ~87% | High | Very Fast | Good |
| Random Forest | ~86% | Moderate | Fast | Good |
| Gradient Boosting | ~86% | Moderate | Medium | Good |

## Key Features

1. **Multi-Objective Optimization**: Balances accuracy, fairness, and energy consumption
2. **Conformal Fairness**: Provides statistical guarantees on fairness metrics
3. **Energy Monitoring**: Tracks computational energy consumption using Green Algorithms approximation
4. **Pareto Optimization**: Finds trade-off solutions across objectives
5. **Surrogate Modeling**: Uses proxy datasets for efficient hyperparameter search

## Installation Requirements

```bash
pip install numpy pandas scikit-learn xgboost catboost lightgbm
```

## Output

Each run generates:
- `cafe_mo_all.csv`: All evaluated configurations
- `cafe_mo_pareto.csv`: Pareto-optimal solutions
- `cafe_mo_log.jsonl`: Detailed optimization log
- Model comparison results when using `run_all_models.py`