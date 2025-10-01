# scripts/plot_results.py
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def _read_runs_for_dataset(root: Path, dataset: str):
    """
    Return dict: { run_name -> { 'all': df_all, 'pareto': df_pareto } } 
    Supports both old format (RF_ADULT folders) and new format (AllModels_ADULT/RF/ folders)
    """
    out = {}
    
    # Try new organized format first (AllModels_DATASET/)
    all_models_dir = root / f"AllModels_{dataset}"
    if all_models_dir.exists() and all_models_dir.is_dir():
        print(f"Found organized structure: {all_models_dir}")
        for model_dir in sorted(all_models_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue
            
            all_csv = model_dir / "cafe_mo_all.csv"
            pareto_csv = model_dir / "cafe_mo_pareto.csv"
            
            if all_csv.exists():
                df_all = pd.read_csv(all_csv)
            else:
                df_all = pd.DataFrame()
            if pareto_csv.exists():
                df_pareto = pd.read_csv(pareto_csv)
            else:
                df_pareto = pd.DataFrame()
            
            model_name = model_dir.name
            out[f"{model_name}_{dataset}"] = {"all": df_all, "pareto": df_pareto, "dir": model_dir}
    
    # Fallback to old format (MODEL_DATASET folders)
    if not out:
        print(f"Looking for old format folders ending with _{dataset}")
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            if not sub.name.endswith(f"_{dataset}"):
                continue
            
            all_csv = sub / "cafe_mo_all.csv"
            pareto_csv = sub / "cafe_mo_pareto.csv"
            
            if all_csv.exists():
                df_all = pd.read_csv(all_csv)
            else:
                df_all = pd.DataFrame()
            if pareto_csv.exists():
                df_pareto = pd.read_csv(pareto_csv)
            else:
                df_pareto = pd.DataFrame()
            
            out[sub.name] = {"all": df_all, "pareto": df_pareto, "dir": sub}
    
    return out

def _nice_name(run_name: str):
    # RF_ADULT -> RF, LOGREG_ADULT -> LOGREG, XGB_ADULT -> XGB
    model_name = run_name.split("_")[0]
    # Map to nicer display names
    name_mapping = {
        'RF': 'Random Forest',
        'LOGREG': 'Logistic Regression', 
        'SVM': 'Support Vector Machine',
        'XGB': 'XGBoost',
        'LGBM': 'LightGBM',
        'CATBOOST': 'CatBoost',
        'GB': 'Gradient Boosting'
    }
    return name_mapping.get(model_name, model_name)

def plot_pareto(root: Path, dataset: str, save: bool = True):
    runs = _read_runs_for_dataset(root, dataset)
    if not runs:
        print(f"No runs found under {root} for dataset {dataset}")
        return
        
    print(f"Found {len(runs)} model runs: {list(runs.keys())}")
    
    # Use better colors for more models
    colors = plt.cm.Set1(np.linspace(0, 1, max(8, len(runs))))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy vs Fairness (DSP)
    for i, (name, blobs) in enumerate(runs.items()):
        pareto = blobs["pareto"]
        if pareto.empty:
            print(f"No Pareto solutions for {name}")
            continue
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        scatter = ax1.scatter(pareto["dsp"], pareto["accuracy"], 
                            s=60, alpha=0.8, c=[color], 
                            marker=marker, label=_nice_name(name),
                            edgecolors='black', linewidth=0.5)
    
    ax1.set_xlabel("Fairness (DSP) ↓", fontsize=12)
    ax1.set_ylabel("Accuracy ↑", fontsize=12)
    ax1.set_title(f"Accuracy vs Fairness Trade-off — {dataset}", fontsize=14)
    ax1.legend(title="Model", loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy vs Accuracy
    for i, (name, blobs) in enumerate(runs.items()):
        pareto = blobs["pareto"]
        if pareto.empty or "kWh" not in pareto.columns:
            continue
            
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        
        ax2.scatter(pareto["kWh"], pareto["accuracy"], 
                   s=60, alpha=0.8, c=[color], 
                   marker=marker, label=_nice_name(name),
                   edgecolors='black', linewidth=0.5)
    
    ax2.set_xlabel("Energy Consumption (kWh) ↓", fontsize=12)
    ax2.set_ylabel("Accuracy ↑", fontsize=12)
    ax2.set_title(f"Accuracy vs Energy Trade-off — {dataset}", fontsize=14)
    ax2.legend(title="Model", loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')  # Log scale for energy
    
    plt.tight_layout()
    
    if save:
        # Save in organized format
        all_models_dir = root / f"AllModels_{dataset}"
        if all_models_dir.exists():
            out = all_models_dir
        else:
            out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.savefig(out / f"{dataset}_pareto_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {out / f'{dataset}_pareto_comparison.png'}")
    plt.show()

def plot_convergence(root: Path, dataset: str, save: bool = True):
    runs = _read_runs_for_dataset(root, dataset)
    if not runs:
        print(f"No runs found under {root} for dataset {dataset}")
        return
    
    # Create subplots for comprehensive convergence analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    
    for i, (name, blobs) in enumerate(runs.items()):
        df = blobs["all"]
        if df.empty:
            continue
            
        model_name = _nice_name(name)
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        df = df.reset_index(drop=True).copy()
        
        # Plot 1: Convergence metric - running best of (err + dsp)
        if "err" in df.columns and "dsp" in df.columns:
            scalar = (df["err"].fillna(1.0) + df["dsp"].fillna(1.0)).values
            best = np.minimum.accumulate(scalar)
            ax1.plot(best, label=model_name, alpha=0.8, 
                    color=color, linestyle=line_style, linewidth=2)
        
        # Plot 2: Pareto set size growth
        if all(col in df.columns for col in ["err", "dsp", "kWh"]):
            size_over_time = []
            partial = []
            for j in range(len(df)):
                partial.append((float(df.loc[j, "err"]), float(df.loc[j, "dsp"]), float(df.loc[j, "kWh"])))
                size_over_time.append(_pareto_size(np.array(partial)))
            ax2.plot(size_over_time, label=model_name, alpha=0.8, 
                    color=color, linestyle=line_style, linewidth=2)
        
        # Plot 3: Accuracy Convergence (if available)
        if "accuracy" in df.columns:
            accuracy_best = np.maximum.accumulate(df["accuracy"].fillna(0.0))
            ax3.plot(accuracy_best, label=model_name, alpha=0.8, 
                    color=color, linestyle=line_style, linewidth=2)
        
        # Plot 4: Energy Efficiency Over Time
        if "kWh" in df.columns:
            # Calculate cumulative average energy
            cum_energy = df["kWh"].fillna(0.0).cumsum()
            iterations = range(1, len(cum_energy) + 1)
            avg_energy = cum_energy / np.array(iterations)
            ax4.plot(avg_energy, label=model_name, alpha=0.8, 
                    color=color, linestyle=line_style, linewidth=2)
    
    # Configure Plot 1: Error + DSP Convergence
    ax1.set_title(f"Convergence — Best(Error + DSP) ↓ | {dataset}", fontsize=14)
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Best Combined Metric So Far", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=10)
    
    # Configure Plot 2: Pareto Set Size
    ax2.set_title(f"Convergence — Pareto Set Size ↑ | {dataset}", fontsize=14)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("# Non-dominated Points", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=10)
    
    # Configure Plot 3: Accuracy Convergence
    ax3.set_title(f"Accuracy Convergence ↑ | {dataset}", fontsize=14)
    ax3.set_xlabel("Iteration", fontsize=12)
    ax3.set_ylabel("Best Accuracy So Far", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=10)
    
    # Configure Plot 4: Energy Efficiency
    ax4.set_title(f"Average Energy per Iteration | {dataset}", fontsize=14)
    ax4.set_xlabel("Iteration", fontsize=12)
    ax4.set_ylabel("Cumulative Avg Energy (kWh)", fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    
    if save:
        # Save in organized format
        all_models_dir = root / f"AllModels_{dataset}"
        if all_models_dir.exists():
            out = all_models_dir
        else:
            out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.savefig(out / f"{dataset}_convergence_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Saved convergence analysis to: {out / f'{dataset}_convergence_analysis.png'}")
    
    plt.show()

def _pareto_size(vals: np.ndarray) -> int:
    # vals: N x 3 (err, dsp, kWh)  -> count non-dominated (minimize all)
    if len(vals) == 0:
        return 0
    nd = np.ones(len(vals), dtype=bool)
    for i in range(len(vals)):
        if not nd[i]:
            continue
        dominated = (np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1))
        nd[dominated] = False
    return int(nd.sum())

def plot_energy_bars(root: Path, dataset: str, save: bool = True):
    runs = _read_runs_for_dataset(root, dataset)
    rows = []
    for name, blobs in runs.items():
        df = blobs["all"]
        if df.empty or "kWh" not in df.columns:
            continue
        rows.append({
            "method": _nice_name(name),
            "total_kWh": float(df["kWh"].sum()),
            "avg_kWh_per_eval": float(df["kWh"].mean()),
            "median_kWh": float(df["kWh"].median()),
            "total_gCO2e": float(df.get("gCO2e", pd.Series([0]*len(df))).sum()),
            "evaluations": len(df),
            "best_accuracy": float(df["accuracy"].max()) if "accuracy" in df.columns else 0.0
        })
    
    if not rows:
        print(f"No energy data found for {dataset}")
        return
        
    agg = pd.DataFrame(rows)
    
    # Create subplots for comprehensive energy analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Total Energy Consumption
    colors = plt.cm.Set3(np.linspace(0, 1, len(agg)))
    bars1 = ax1.bar(agg["method"], agg["total_kWh"], color=colors, width=0.6)
    ax1.set_ylabel("Total Energy (kWh)", fontsize=12)
    ax1.set_title(f"Total Energy Consumption — {dataset}", fontsize=14)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, agg["total_kWh"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Average Energy per Evaluation
    bars2 = ax2.bar(agg["method"], agg["avg_kWh_per_eval"], color=colors, width=0.6)
    ax2.set_ylabel("Average Energy per Evaluation (kWh)", fontsize=12)
    ax2.set_title(f"Energy Efficiency — {dataset}", fontsize=14)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: CO2 Emissions
    if agg["total_gCO2e"].sum() > 0:
        bars3 = ax3.bar(agg["method"], agg["total_gCO2e"], color=colors, width=0.6)
        ax3.set_ylabel("Total CO2 Emissions (g)", fontsize=12)
        ax3.set_title(f"Carbon Footprint — {dataset}", fontsize=14)
        ax3.grid(axis="y", alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Energy vs Performance Trade-off
    scatter = ax4.scatter(agg["total_kWh"], agg["best_accuracy"], 
                         c=range(len(agg)), s=100, alpha=0.7, cmap='viridis')
    
    # Add model labels
    for i, row in agg.iterrows():
        ax4.annotate(row["method"], (row["total_kWh"], row["best_accuracy"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel("Total Energy (kWh)", fontsize=12)
    ax4.set_ylabel("Best Accuracy", fontsize=12)
    ax4.set_title(f"Energy vs Performance Trade-off — {dataset}", fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        # Save in organized format
        all_models_dir = root / f"AllModels_{dataset}"
        if all_models_dir.exists():
            out = all_models_dir
        else:
            out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.savefig(out / f"{dataset}_energy_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Saved energy analysis to: {out / f'{dataset}_energy_analysis.png'}")
    
    # Print summary table
    print(f"\nEnergy Summary for {dataset}:")
    print("=" * 80)
    summary_df = agg[["method", "total_kWh", "avg_kWh_per_eval", "best_accuracy", "evaluations"]].copy()
    summary_df.columns = ["Model", "Total kWh", "Avg kWh/eval", "Best Accuracy", "Evaluations"]
    print(summary_df.to_string(index=False, float_format='%.6f'))
    
    plt.show()

def plot_3d_scatter(root: Path, run_dir: str, save: bool = True):
    """
    For a SINGLE run folder (e.g., outputs/RF_ADULT/) show 3D scatter of
    Accuracy (↑), DSP (↓), kWh (↓).
    """
    run_path = Path(run_dir)
    df = pd.read_csv(run_path / "cafe_mo_all.csv")
    if df.empty:
        print("No data for 3D plot")
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    xs = df["dsp"]; ys = df["accuracy"]; zs = df["kWh"]
    ax.scatter(xs, ys, zs, s=25, depthshade=True)
    ax.set_xlabel("DSP ↓"); ax.set_ylabel("Accuracy ↑"); ax.set_zlabel("kWh ↓")
    ax.set_title(f"3D trade-off — {run_path.name}")
    if save:
        out = run_path
        plt.tight_layout()
        plt.savefig(out / f"{run_path.name}_3D.png", dpi=200)
    plt.show()

def plot_model_comparison(root: Path, dataset: str, save: bool = True):
    """Create comprehensive model comparison across all metrics"""
    runs = _read_runs_for_dataset(root, dataset)
    if not runs:
        print(f"No runs found for {dataset}")
        return
    
    # Collect summary statistics for each model
    model_stats = []
    for name, blobs in runs.items():
        df = blobs["all"]
        if df.empty:
            continue
            
        stats = {
            "Model": _nice_name(name),
            "Best_Accuracy": df["accuracy"].max() if "accuracy" in df.columns else 0.0,
            "Best_DSP": df["dsp"].min() if "dsp" in df.columns else 1.0,
            "Total_Energy": df["kWh"].sum() if "kWh" in df.columns else 0.0,
            "Avg_Energy": df["kWh"].mean() if "kWh" in df.columns else 0.0,
            "Evaluations": len(df),
            "Efficiency_Score": (df["accuracy"].max() if "accuracy" in df.columns else 0.0) / 
                              (df["kWh"].mean() if "kWh" in df.columns and df["kWh"].mean() > 0 else 1.0)
        }
        model_stats.append(stats)
    
    if not model_stats:
        print(f"No valid model data found for {dataset}")
        return
    
    df_stats = pd.DataFrame(model_stats)
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance Bar Chart
    if len(df_stats) > 1:
        x_pos = range(len(df_stats))
        width = 0.35
        
        # Normalize accuracy for comparison
        norm_acc = df_stats["Best_Accuracy"] / df_stats["Best_Accuracy"].max() if df_stats["Best_Accuracy"].max() > 0 else df_stats["Best_Accuracy"]
        norm_eff = df_stats["Efficiency_Score"] / df_stats["Efficiency_Score"].max() if df_stats["Efficiency_Score"].max() > 0 else df_stats["Efficiency_Score"]
        
        ax1.bar([x - width/2 for x in x_pos], norm_acc, width, 
               label="Normalized Accuracy", alpha=0.8, color='skyblue')
        ax1.bar([x + width/2 for x in x_pos], norm_eff, width, 
               label="Normalized Efficiency", alpha=0.8, color='orange')
        ax1.set_xlabel("Models", fontsize=12)
        ax1.set_ylabel("Normalized Score", fontsize=12)
        ax1.set_title(f"Performance Comparison — {dataset}", fontsize=14)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_stats["Model"], rotation=45)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Energy Consumption Comparison
    bars = ax2.bar(df_stats["Model"], df_stats["Total_Energy"], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(df_stats))))
    ax2.set_xlabel("Models", fontsize=12)
    ax2.set_ylabel("Total Energy (kWh)", fontsize=12)
    ax2.set_title(f"Energy Consumption — {dataset}", fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, df_stats["Total_Energy"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01, 
                f'{value:.6f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Accuracy vs Energy Scatter
    scatter = ax3.scatter(df_stats["Total_Energy"], df_stats["Best_Accuracy"], 
                         s=100, alpha=0.7, c=range(len(df_stats)), cmap='plasma')
    
    for i, row in df_stats.iterrows():
        ax3.annotate(row["Model"], (row["Total_Energy"], row["Best_Accuracy"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel("Total Energy (kWh)", fontsize=12)
    ax3.set_ylabel("Best Accuracy", fontsize=12)
    ax3.set_title(f"Efficiency Trade-off — {dataset}", fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Evaluations Comparison
    bars4 = ax4.bar(df_stats["Model"], df_stats["Evaluations"], 
                    color=plt.cm.plasma(np.linspace(0, 1, len(df_stats))))
    ax4.set_xlabel("Models", fontsize=12)
    ax4.set_ylabel("Number of Evaluations", fontsize=12)
    ax4.set_title(f"Optimization Iterations — {dataset}", fontsize=14)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        # Save in organized format
        all_models_dir = root / f"AllModels_{dataset}"
        if all_models_dir.exists():
            out = all_models_dir
        else:
            out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.savefig(out / f"{dataset}_model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to: {out / f'{dataset}_model_comparison.png'}")
    
    # Print summary table
    print(f"\nModel Comparison Summary for {dataset}:")
    print("=" * 100)
    display_df = df_stats.copy()
    display_df = display_df.round(6)
    print(display_df.to_string(index=False))
    
    plt.show()
    return df_stats


def main():
    ap = argparse.ArgumentParser(description="Generate comprehensive plots for CAFE-MO results")
    ap.add_argument("--root", default="outputs", help="outputs root directory")
    ap.add_argument("--dataset", required=True, choices=["ADULT","COMPAS","GERMANCREDIT","LAWSCHOOL"])
    ap.add_argument("--plots", nargs="+", default=["pareto","convergence","energy","comparison"],
                    choices=["pareto","convergence","energy","3d","comparison","all"])
    ap.add_argument("--run_dir", default="", help="for plots=3d: path to a single run dir (e.g., outputs/RF_ADULT)")
    ap.add_argument("--all-datasets", action="store_true",
                   help="Generate plots for all available datasets")
    args = ap.parse_args()

    root = Path(args.root)
    
    # Determine datasets to process
    datasets = [args.dataset]
    if args.all_datasets:
        datasets = []
        # Look for AllModels_* directories
        for item in root.iterdir():
            if item.is_dir() and item.name.startswith("AllModels_"):
                dataset_name = item.name.replace("AllModels_", "")
                datasets.append(dataset_name)
        if not datasets:
            print("No AllModels_* directories found. Using provided dataset.")
            datasets = [args.dataset]
    
    # Handle "all" plots option
    if "all" in args.plots:
        args.plots = ["pareto", "convergence", "energy", "comparison", "3d"]
    
    print(f"Generating plots for datasets: {datasets}")
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*60}")
        
        try:
            if "pareto" in args.plots:
                plot_pareto(root, dataset)
            if "convergence" in args.plots:
                plot_convergence(root, dataset)
            if "energy" in args.plots:
                plot_energy_bars(root, dataset)
            if "comparison" in args.plots:
                plot_model_comparison(root, dataset)
            if "3d" in args.plots:
                if not args.run_dir:
                    print("Please pass --run_dir for 3d plot.")
                else:
                    plot_3d_scatter(root, args.run_dir)
        except Exception as e:
            print(f"Error generating plots for {dataset}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
