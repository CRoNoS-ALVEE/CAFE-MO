# CAFE-MO Enhanced Plotting Guide

This guide explains how to use the enhanced plotting system in CAFE-MO to visualize and compare model performance across multiple dimensions.

## Quick Start

```bash
# Generate all plots for a single dataset
python scripts/plot_results.py --root outputs --dataset ADULT --plots all

# Generate specific plot types
python scripts/plot_results.py --root outputs --dataset ADULT --plots pareto energy comparison

# Generate plots for all available datasets
python scripts/plot_results.py --root outputs --dataset ADULT --all-datasets --plots comparison
```

## Available Plot Types

### 1. Pareto Front Analysis (`pareto`)
- **Dual subplots**: Accuracy vs Fairness (DSP) and Accuracy vs Energy
- **Multi-model comparison**: Shows all 7 models on the same plots
- **Interactive legend**: Click to hide/show models
- **Enhanced styling**: Better colors, markers, and annotations

**Output**: `{dataset}_pareto_comparison.png`

### 2. Energy Analysis (`energy`)
- **Four comprehensive panels**:
  - Total energy consumption by model
  - Average energy per evaluation (efficiency)
  - CO2 emissions comparison
  - Energy vs performance trade-off scatter plot
- **Summary table**: Detailed energy metrics printed to console
- **Value annotations**: Energy values displayed on bar charts

**Output**: `{dataset}_energy_analysis.png`

### 3. Convergence Analysis (`convergence`)
- **Four tracking panels**:
  - Combined error + DSP convergence
  - Pareto set size growth over iterations
  - Best accuracy convergence
  - Cumulative average energy per iteration
- **Multiple line styles**: Different colors and line patterns for each model
- **Grid and legends**: Enhanced readability

**Output**: `{dataset}_convergence_analysis.png`

### 4. Model Comparison (`comparison`) - **NEW!**
- **Comprehensive model rankings**:
  - Normalized performance comparison
  - Energy consumption comparison
  - Efficiency trade-off scatter plot
  - Number of evaluations comparison
- **Summary statistics table**: Complete model performance metrics
- **Efficiency score**: Accuracy per unit energy calculation

**Output**: `{dataset}_model_comparison.png`

### 5. 3D Scatter (`3d`)
- **Three-dimensional visualization**: Accuracy, Fairness, Energy
- **Single run analysis**: Detailed view of optimization trajectory
- **Requires `--run_dir`**: Specify path to specific model run

**Usage**: `--plots 3d --run_dir outputs/RF_ADULT`

## Command Line Options

```bash
python scripts/plot_results.py [OPTIONS]

Required Arguments:
  --dataset {ADULT,COMPAS,GERMANCREDIT,LAWSCHOOL}
                        Dataset to analyze

Optional Arguments:
  --root OUTPUT_DIR     Root directory containing results (default: outputs)
  --plots {pareto,convergence,energy,comparison,3d,all} [{pareto,convergence,energy,comparison,3d,all} ...]
                        Plot types to generate (default: pareto,convergence,energy,comparison)
  --all-datasets        Generate plots for all available datasets
  --run_dir RUN_DIR     For 3d plots: path to single run directory
```

## Output Organization

The enhanced plotting system automatically detects your output structure:

### New Organized Structure (Recommended)
```
outputs/
├── AllModels_ADULT/
│   ├── ADULT_pareto_comparison.png
│   ├── ADULT_energy_analysis.png
│   ├── ADULT_convergence_analysis.png
│   └── ADULT_model_comparison.png
├── AllModels_COMPAS/
│   └── ...
```

### Legacy Structure (Also Supported)
```
outputs/
├── plots_ADULT/
│   ├── ADULT_pareto_comparison.png
│   └── ...
```

## Model Support

The enhanced plotting system supports all 7 integrated models:

1. **Random Forest** (RF)
2. **Logistic Regression** (LogReg)
3. **Support Vector Machine** (SVM)
4. **XGBoost** (XGB)
5. **CatBoost** (CatBoost)
6. **LightGBM** (LGBM)
7. **Gradient Boosting** (GB)

## Performance Insights

The plotting system provides insights into:

- **Accuracy**: Best classification performance achieved
- **Fairness**: Demographic Statistical Parity (DSP) - lower is better
- **Energy Efficiency**: kWh consumed per evaluation
- **Convergence Speed**: How quickly models reach optimal solutions
- **Trade-offs**: Relationships between accuracy, fairness, and energy

## Example Output Interpretation

### Energy Summary Table
```
Model Comparison Summary for ADULT:
                 Model  Best_Accuracy  Best_DSP  Total_Energy  Avg_Energy  Evaluations  Efficiency_Score
              LightGBM       0.871208  0.000000      0.000031    0.000004            7     199128.588891
   Logistic Regression       0.849163  0.075602      0.000012    0.000002            7     505599.622648
```

**Key Insights**:
- **LightGBM**: Highest accuracy (87.12%), perfect fairness, very energy efficient
- **Logistic Regression**: Lower accuracy but extremely energy efficient
- **Efficiency Score**: Accuracy per unit energy (higher is better)

## Tips for Best Results

1. **Run all models first**: Use `python scripts/run_all_models.py` before plotting
2. **Use --all-datasets**: Generate comprehensive comparisons across all datasets
3. **Check energy patterns**: Look for models that balance accuracy with low energy consumption
4. **Compare convergence**: Identify which models reach optimal solutions faster
5. **Save plots**: All plots are automatically saved in high resolution (300 DPI)

## Troubleshooting

- **No plots generated**: Ensure model runs exist in the outputs directory
- **Missing models**: Some models may be missing if dependencies aren't installed
- **Empty plots**: Check that CSV files contain expected columns (accuracy, dsp, kWh)
- **3D plot errors**: Ensure `--run_dir` points to a valid single model run directory

## Advanced Usage

```bash
# Generate plots for multiple datasets at once
python scripts/plot_results.py --root outputs --dataset ADULT --all-datasets --plots all

# Focus on energy analysis across all datasets
python scripts/plot_results.py --root outputs --dataset ADULT --all-datasets --plots energy

# Quick comparison of model performance
python scripts/plot_results.py --root outputs --dataset ADULT --plots comparison
```

The enhanced plotting system provides comprehensive visualization capabilities to help you understand the multi-objective trade-offs in your CAFE-MO optimization results!