from pathlib import Path
import argparse
import sys
from catboost import CatBoostClassifier

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

def make_catboost(params):
    return CatBoostClassifier(
        iterations=int(params.get("iterations", 100)),
        depth=int(params.get("depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bylevel=float(params.get("colsample_bylevel", 1.0)),
        random_seed=42,
        thread_count=-1,
        verbose=False,  # Suppress output during training
        allow_writing_files=False  # Prevent creating temporary files
    )

def sample_catboost(rng):
    return {
        "iterations": int(rng.randint(50, 300)),
        "depth": int(rng.randint(4, 10)),
        "learning_rate": float(rng.uniform(0.01, 0.3)),
        "l2_leaf_reg": float(10 ** rng.uniform(-1, 2)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
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
    out = Path(args.outdir)/("CATBOOST_"+args.dataset)
    opt = CafeMOOptimizer(cfg, model_fn=make_catboost, sample_params_fn=sample_catboost)
    df, pareto = opt.run(out)
    print("Saved to:", out)
    print(pareto.head())

if __name__ == "__main__":
    main()