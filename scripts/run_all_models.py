from pathlib import Path
import argparse
import sys
import pandas as pd
import time

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

# Import all model functions
from run_rf import make_rf, sample_rf
from run_logreg import make_logreg, sample_logreg
from run_svm import make_svm, sample_svm
from run_xgb import make_xgb, sample_xgb
from run_catboost import make_catboost, sample_catboost
from run_lgbm import make_lgbm, sample_lgbm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ADULT","COMPAS","GERMANCREDIT","LAWSCHOOL"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--models", nargs="+", default=["RF", "LOGREG", "SVM", "XGB", "CATBOOST", "LGBM"],
                   help="Models to run. Choose from: RF, LOGREG, SVM, XGB, CATBOOST, LGBM")
    args = p.parse_args()

    base = Path("DataSets")
    paths = {
        "ADULT": base/"ADULT_full.csv",
        "COMPAS": base/"COMPAS_full.csv",
        "GERMANCREDIT": base/"GERMANCREDIT_full.csv",
        "LAWSCHOOL": base/"LAWSCHOOLADMISSIONS_full.csv",
    }

    # Model configurations
    model_configs = {
        "RF": (make_rf, sample_rf),
        "LOGREG": (make_logreg, sample_logreg),
        "SVM": (make_svm, sample_svm),
        "XGB": (make_xgb, sample_xgb),
        "CATBOOST": (make_catboost, sample_catboost),
        "LGBM": (make_lgbm, sample_lgbm),
    }

    cfg = Config(
        dataset_name=args.dataset,
        dataset_path=paths[args.dataset],
        rounds=args.rounds,
        test_size=args.test_size,
        tau=args.tau,
        alpha=args.alpha,
    )

    results_summary = []
    
    for model_name in args.models:
        if model_name not in model_configs:
            print(f"Unknown model: {model_name}. Skipping...")
            continue
            
        print(f"\n{'='*50}")
        print(f"Running {model_name} on {args.dataset} dataset...")
        print(f"{'='*50}")
        
        model_fn, sample_fn = model_configs[model_name]
        out = Path(args.outdir)/(f"{model_name}_{args.dataset}")
        
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

    # Save comparison results
    if results_summary:
        comparison_df = pd.DataFrame(results_summary)
        comparison_path = Path(args.outdir)/f"model_comparison_{args.dataset}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(comparison_df.to_string(index=False))
        print(f"\nComparison results saved to: {comparison_path}")

if __name__ == "__main__":
    main()