from pathlib import Path
import argparse
import sys
from sklearn.ensemble import GradientBoostingClassifier

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

def make_gb(params):
    return GradientBoostingClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 1.0)),
        max_features=params.get("max_features", None),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        random_state=42
    )

def sample_gb(rng):
    max_features_choices = [None, "sqrt", "log2"]
    return {
        "n_estimators": int(rng.randint(50, 200)),
        "max_depth": int(rng.randint(2, 8)),
        "learning_rate": float(rng.uniform(0.01, 0.3)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "max_features": rng.choice(max_features_choices),
        "min_samples_split": int(rng.randint(2, 10)),
        "min_samples_leaf": int(rng.randint(1, 5)),
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
    out = Path(args.outdir)/("GB_"+args.dataset)
    opt = CafeMOOptimizer(cfg, model_fn=make_gb, sample_params_fn=sample_gb)
    df, pareto = opt.run(out)
    print("Saved to:", out)
    print(pareto.head())

if __name__ == "__main__":
    main()