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
    where run_name is e.g. RF_ADULT, LOGREG_ADULT, SVM_ADULT.
    """
    out = {}
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
    # RF_ADULT -> RF, LOGREG_ADULT -> LOGREG
    return run_name.split("_")[0]

def plot_pareto(root: Path, dataset: str, save: bool = True):
    runs = _read_runs_for_dataset(root, dataset)
    if not runs:
        print(f"No runs found under {root} for dataset {dataset}")
        return
    fig, ax = plt.subplots(figsize=(7,5))
    for name, blobs in runs.items():
        pareto = blobs["pareto"]
        if pareto.empty:
            continue
        # X = fairness (DSP), Y = accuracy; color = kWh
        sc = ax.scatter(pareto["dsp"], pareto["accuracy"], s=40, alpha=0.85,
                        label=_nice_name(name), c=pareto.get("kWh", pd.Series([0]*len(pareto))))
    ax.set_xlabel("Fairness (DSP) ↓")
    ax.set_ylabel("Accuracy ↑")
    ax.set_title(f"Pareto Fronts — {dataset}")
    ax.legend(title="Method", loc="best")
    ax.grid(True, alpha=0.3)
    if save:
        out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(out / f"{dataset}_pareto.png", dpi=200)
    plt.show()

def plot_convergence(root: Path, dataset: str, save: bool = True):
    runs = _read_runs_for_dataset(root, dataset)
    if not runs:
        print(f"No runs found under {root} for dataset {dataset}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    ax1, ax2 = axes
    for name, blobs in runs.items():
        df = blobs["all"]
        if df.empty or "err" not in df.columns:
            continue
        df = df.reset_index(drop=True).copy()
        # Convergence metric 1: running best of (err + dsp)
        scalar = (df["err"].fillna(1.0) + df["dsp"].fillna(1.0)).values
        best = np.minimum.accumulate(scalar)
        ax1.plot(best, label=_nice_name(name))
        # Convergence metric 2: Pareto set size growth
        size_over_time = []
        partial = []
        for i in range(len(df)):
            partial.append((float(df.loc[i, "err"]), float(df.loc[i, "dsp"]), float(df.loc[i, "kWh"])))
            size_over_time.append(_pareto_size(np.array(partial)))
        ax2.plot(size_over_time, label=_nice_name(name))
    ax1.set_title(f"Convergence — best(err + DSP) ↓ | {dataset}")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Best so far")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax2.set_title(f"Convergence — Pareto set size ↑ | {dataset}")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("# Non-dominated points")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    if save:
        out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(out / f"{dataset}_convergence.png", dpi=200)
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
            "total_gCO2e": float(df.get("gCO2e", pd.Series([0]*len(df))).sum())
        })
    if not rows:
        print(f"No energy data found for {dataset}")
        return
    agg = pd.DataFrame(rows).groupby("method", as_index=False).sum(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(agg["method"], agg["total_kWh"], width=0.6)
    ax.set_ylabel("Total energy (kWh)")
    ax.set_title(f"Energy comparison by method — {dataset}")
    ax.grid(axis="y", alpha=0.3)
    if save:
        out = root / f"plots_{dataset}"
        out.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(out / f"{dataset}_energy.png", dpi=200)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs", help="outputs root directory")
    ap.add_argument("--dataset", required=True, choices=["ADULT","COMPAS","GERMANCREDIT","LAWSCHOOL"])
    ap.add_argument("--plots", nargs="+", default=["pareto","convergence","energy"],
                    choices=["pareto","convergence","energy","3d"])
    ap.add_argument("--run_dir", default="", help="for plots=3d: path to a single run dir (e.g., outputs/RF_ADULT)")
    args = ap.parse_args()

    root = Path(args.root)

    if "pareto" in args.plots:
        plot_pareto(root, args.dataset)
    if "convergence" in args.plots:
        plot_convergence(root, args.dataset)
    if "energy" in args.plots:
        plot_energy_bars(root, args.dataset)
    if "3d" in args.plots:
        if not args.run_dir:
            print("Please pass --run_dir for 3d plot.")
        else:
            plot_3d_scatter(root, args.run_dir)

if __name__ == "__main__":
    main()
