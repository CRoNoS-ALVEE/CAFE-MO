from pathlib import Path
import argparse
import sys
import pandas as pd
import time

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

# Import model functions with error handling for missing dependencies
available_models = {}

# Always available models (use basic sklearn)
try:
    from run_rf import make_rf, sample_rf
    available_models["RF"] = (make_rf, sample_rf)
except ImportError as e:
    print(f"Warning: Could not import Random Forest: {e}")

try:
    from run_logreg import make_logreg, sample_logreg
    available_models["LOGREG"] = (make_logreg, sample_logreg)
except ImportError as e:
    print(f"Warning: Could not import Logistic Regression: {e}")

try:
    from run_svm import make_svm, sample_svm
    available_models["SVM"] = (make_svm, sample_svm)
except ImportError as e:
    print(f"Warning: Could not import SVM: {e}")

# Optional models (require additional packages)
try:
    from run_xgb import make_xgb, sample_xgb
    available_models["XGB"] = (make_xgb, sample_xgb)
except ImportError:
    print("Info: XGBoost not available (run: pip install xgboost)")

try:
    from run_catboost import make_catboost, sample_catboost
    available_models["CATBOOST"] = (make_catboost, sample_catboost)
except ImportError:
    print("Info: CatBoost not available (run: pip install catboost)")

try:
    from run_lgbm import make_lgbm, sample_lgbm
    available_models["LGBM"] = (make_lgbm, sample_lgbm)
except ImportError:
    print("Info: LightGBM not available (run: pip install lightgbm)")

try:
    from run_gb import make_gb, sample_gb
    available_models["GB"] = (make_gb, sample_gb)
except ImportError:
    print("Info: Gradient Boosting not available")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ADULT","COMPAS","GERMANCREDIT","LAWSCHOOL"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--models", nargs="+", default=list(available_models.keys()),
                   help=f"Models to run. Available: {list(available_models.keys())}")
    args = p.parse_args()

    base = Path("DataSets")
    paths = {
        "ADULT": base/"ADULT_full.csv",
        "COMPAS": base/"COMPAS_full.csv",
        "GERMANCREDIT": base/"GERMANCREDIT_full.csv",
        "LAWSCHOOL": base/"LAWSCHOOLADMISSIONS_full.csv",
    }

    # Use only available model configurations
    model_configs = available_models

    cfg = Config(
        dataset_name=args.dataset,
        dataset_path=paths[args.dataset],
        rounds=args.rounds,
        test_size=args.test_size,
        tau=args.tau,
        alpha=args.alpha,
    )

    results_summary = []
    
    print(f"\nAvailable models: {list(available_models.keys())}")
    print(f"Selected models: {args.models}")
    
    for model_name in args.models:
        if model_name not in model_configs:
            print(f"Unknown model: {model_name}. Skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"Running {model_name} on {args.dataset} dataset...")
        print(f"{'='*50}")
        
        model_fn, sample_fn = model_configs[model_name]
        # Create separate folder for each model
        model_output_dir = Path(args.outdir)/f"AllModels_{args.dataset}"/model_name
        out = model_output_dir
        
        start_time = time.time()
        try:
            opt = CafeMOOptimizer(cfg, model_fn=model_fn, sample_params_fn=sample_fn)
            df, pareto = opt.run(out)
            end_time = time.time()
            
            if not pareto.empty:
                # Get best results (Pareto front)
                best_acc = pareto['accuracy'].max()
                best_dsp = pareto['dsp'].min()
                best_energy = pareto['kWh'].min()
                
                results_summary.append({
                    'Model': model_name,
                    'Dataset': args.dataset,
                    'Best_Accuracy': best_acc,
                    'Best_DSP': best_dsp,
                    'Best_Energy_kWh': best_energy,
                    'Total_Time_sec': end_time - start_time,
                    'Pareto_Solutions': len(pareto)
                })
                
                print(f"Completed {model_name}!")
                print(f"Best accuracy: {best_acc:.4f}")
                print(f"Best fairness (DSP): {best_dsp:.4f}")
                print(f"Best energy: {best_energy:.6f} kWh")
                print(f"Pareto solutions found: {len(pareto)}")
                print(f"Results saved to: {out}")
            else:
                print(f"No Pareto solutions found for {model_name}")
                
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            end_time = time.time()
            results_summary.append({
                'Model': model_name,
                'Dataset': args.dataset,
                'Best_Accuracy': None,
                'Best_DSP': None,
                'Best_Energy_kWh': None,
                'Total_Time_sec': end_time - start_time,
                'Pareto_Solutions': 0,
                'Error': str(e)
            })

    # Save comparison results in the main AllModels folder
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        main_output_dir = Path(args.outdir)/f"AllModels_{args.dataset}"
        main_output_dir.mkdir(parents=True, exist_ok=True)
        comparison_path = main_output_dir/f"model_comparison_{args.dataset}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        print(f"\nComparison results saved to: {comparison_path}")

if __name__ == "__main__":
    main()