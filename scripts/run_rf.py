from pathlib import Path
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

def make_rf(params):
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators",200)),
        max_depth=int(params.get("max_depth",10)) if params.get("max_depth",None) is not None else None,
        min_samples_split=int(params.get("min_samples_split",2)),
        n_jobs=-1, random_state=0
    )

def sample_rf(rng):
    return {
        "n_estimators": int(rng.randint(100, 400)),
        "max_depth": int(rng.randint(3, 20)),
        "min_samples_split": int(rng.randint(2, 10)),
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
    out = Path(args.outdir)/("RF_"+args.dataset)
    opt = CafeMOOptimizer(cfg, model_fn=make_rf, sample_params_fn=sample_rf)
    df, pareto = opt.run(out)
    print("Saved to:", out)
    print(pareto.head())

if __name__ == "__main__":
    main()
