from pathlib import Path
import argparse
import sys
from lightgbm import LGBMClassifier

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

def make_lgbm(params):
    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", -1)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 0.0)),
        num_leaves=int(params.get("num_leaves", 31)),
        min_child_samples=int(params.get("min_child_samples", 20)),
        random_state=42,
        n_jobs=-1,
        verbosity=-1  # Suppress output
    )

def sample_lgbm(rng):
    return {
        "n_estimators": int(rng.randint(50, 300)),
        "max_depth": int(rng.randint(3, 15)) if rng.rand() > 0.3 else -1,  # -1 means no limit
        "learning_rate": float(rng.uniform(0.01, 0.3)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(10 ** rng.uniform(-3, 1)),
        "reg_lambda": float(10 ** rng.uniform(-3, 1)),
        "num_leaves": int(rng.randint(10, 100)),
        "min_child_samples": int(rng.randint(5, 50)),
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
    out = Path(args.outdir)/("LGBM_"+args.dataset)
    opt = CafeMOOptimizer(cfg, model_fn=make_lgbm, sample_params_fn=sample_lgbm)
    df, pareto = opt.run(out)
    print("Saved to:", out)
    print(pareto.head())

if __name__ == "__main__":
    main()