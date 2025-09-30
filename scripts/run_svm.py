from pathlib import Path
import argparse
import sys
from sklearn.svm import SVC

# Add parent directory to path to import src module
sys.path.append(str(Path(__file__).parent.parent))
from src.optimizer import Config, CafeMOOptimizer

def make_svm(params):
    return SVC(kernel="rbf", C=float(params.get("C",1.0)), gamma=params.get("gamma","scale"))

def sample_svm(rng):
    return {"C": float(10 ** rng.uniform(-2, 2)), "gamma": "scale"}

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

    cfg = Config(dataset_name=args.dataset, dataset_path=paths[args.dataset],
                 rounds=args.rounds, test_size=args.test_size, tau=args.tau, alpha=args.alpha)
    out = Path(args.outdir)/("SVM_"+args.dataset)
    opt = CafeMOOptimizer(cfg, model_fn=make_svm, sample_params_fn=sample_svm)
    df, pareto = opt.run(out)
    print("Saved to:", out)
    print(pareto.head())

if __name__ == "__main__":
    main()
