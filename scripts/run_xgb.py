from pathlib import Path
import argparse
import sys
from xgboost import XGBClassifier

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

def make_xgb(params):
    return XGBClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'  # Suppress warning
    )

def sample_xgb(rng):
    return {
        "n_estimators": int(rng.randint(50, 300)),
        "max_depth": int(rng.randint(3, 10)),
        "learning_rate": float(rng.uniform(0.01, 0.3)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(10 ** rng.uniform(-3, 1)),
        "reg_lambda": float(10 ** rng.uniform(-3, 2)),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ADULT","COMPAS","GERMANCREDIT","LAWSCHOOL"])
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--outdir", default="outputs")
    args = p.parse_args()

    base = Path("DataSets")
    paths = {
        "ADULT": base/"ADULT_full.csv",
        "COMPAS": base/"COMPAS_full.csv",
        "GERMANCREDIT": base/"GERMANCREDIT_full.csv",
        "LAWSCHOOL": base/"LAWSCHOOLADMISSIONS_full.csv",
    }

    cfg = Config(
        dataset_name=args.dataset,
        dataset_path=paths[args.dataset],
        rounds=args.rounds,
        test_size=args.test_size,
        tau=args.tau,
        alpha=args.alpha,
    )
    out = Path(args.outdir)/("XGB_"+args.dataset)
    opt = CafeMOOptimizer(cfg, model_fn=make_xgb, sample_params_fn=sample_xgb)
    df, pareto = opt.run(out)
    print("Saved to:", out)
    print(pareto.head())

if __name__ == "__main__":
    main()