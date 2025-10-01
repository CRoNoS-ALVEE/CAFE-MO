# CAFE-MO Output Organization Guide

## 📁 Organized Output Structure

The `run_all_models.py` script now creates a well-organized folder structure for each dataset and model combination:

### New Folder Structure
```
outputs/
└── AllModels_{DATASET}/
    ├── RF/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    ├── XGB/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    ├── LGBM/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    ├── CATBOOST/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    ├── LOGREG/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    ├── SVM/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    ├── GB/
    │   ├── cafe_mo_all.csv
    │   ├── cafe_mo_pareto.csv
    │   └── cafe_mo_log.jsonl
    └── model_comparison_{DATASET}.csv
```

### Example for ADULT Dataset
```
outputs/AllModels_ADULT/
├── RF/                    # Random Forest results
├── XGB/                   # XGBoost results  
├── LGBM/                  # LightGBM results
├── CATBOOST/              # CatBoost results
├── LOGREG/                # Logistic Regression results
├── SVM/                   # Support Vector Machine results
├── GB/                    # Gradient Boosting results
└── model_comparison_ADULT.csv  # Summary comparison
```

## 🚀 Usage Examples

### Run All Available Models
```bash
python scripts\run_all_models.py --dataset ADULT --rounds 10
```

### Run Specific Models
```bash
python scripts\run_all_models.py --dataset COMPAS --rounds 15 --models XGB LGBM RF
```

### Run with Custom Parameters
```bash
python scripts\run_all_models.py --dataset GERMANCREDIT --rounds 20 --tau 0.15 --alpha 0.05
```

## 📊 Output Files Explained

### Individual Model Folders
Each model gets its own folder containing:
- **`cafe_mo_all.csv`**: All evaluated hyperparameter configurations
- **`cafe_mo_pareto.csv`**: Pareto-optimal solutions (best trade-offs)
- **`cafe_mo_log.jsonl`**: Detailed optimization log with timestamps

### Comparison Summary
- **`model_comparison_{DATASET}.csv`**: Side-by-side performance comparison of all models

## ✨ Key Improvements

### 1. **Error Handling**
- Gracefully handles missing dependencies (XGBoost, CatBoost, LightGBM)
- Only runs models that are available
- Shows informative messages for missing packages

### 2. **Organized Output**
- Each model gets its own dedicated folder
- Clear hierarchical structure by dataset
- Centralized comparison results

### 3. **Flexibility**
- Select specific models to run
- Automatic detection of available models
- Informative progress reporting

## 📈 Sample Results

### Model Performance Comparison (ADULT Dataset, 5 rounds)
| Model    | Best Accuracy | Best DSP | Best Energy (kWh) | Pareto Solutions |
|----------|---------------|----------|-------------------|------------------|
| **LGBM** | 0.8712       | 0.0000   | 0.000001         | 3                |
| CATBOOST | 0.8654       | 0.0470   | 0.000003         | 5                |
| GB       | 0.8634       | 0.0000   | 0.000002         | 7                |
| XGB      | 0.8624       | 0.0000   | 0.000013         | 3                |
| RF       | 0.8561       | 0.0696   | 0.000010         | 4                |
| SVM      | 0.8507       | 0.0000   | 0.000001         | 3                |
| LOGREG   | 0.8478       | 0.0756   | 0.000000         | 6                |

### Key Insights
- **Best Accuracy**: LightGBM (87.12%)
- **Best Fairness**: Multiple models achieve perfect fairness (DSP = 0.0000)
- **Best Energy Efficiency**: Logistic Regression (lowest energy consumption)
- **Most Robust**: Gradient Boosting (7 Pareto solutions)

## 🔧 Troubleshooting

### Missing Packages
If you see warnings about missing models:
```bash
# Install missing packages
pip install xgboost catboost lightgbm

# Or in virtual environment
.\.venv\Scripts\pip.exe install xgboost catboost lightgbm
```

### Virtual Environment Issues
Make sure to activate your virtual environment:
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Then run the script
python scripts\run_all_models.py --dataset ADULT --rounds 10
```

## 📝 Benefits of New Structure

1. **Easy Comparison**: All results for a dataset in one place
2. **Clean Organization**: No file name conflicts between models
3. **Scalable**: Easy to add new models or datasets
4. **Professional**: Publication-ready result organization
5. **Accessible**: Clear folder structure for analysis