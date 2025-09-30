from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, Optional, List
import numpy as np, pandas as pd, time, json, warnings, hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ---------- metrics ----------
def differential_statistical_parity(y_pred, s):
    s = pd.Series(s).values
    y_pred = pd.Series(y_pred).values
    groups = pd.unique(s)
    if len(groups) != 2:
        top2 = pd.Series(s).value_counts().index[:2]
        s = np.where(np.isin(s, top2), s, top2[1])
        groups = pd.unique(s)
    g1, g2 = groups[0], groups[1]
    p1 = y_pred[s == g1].mean() if np.any(s == g1) else 0.0
    p2 = y_pred[s == g2].mean() if np.any(s == g2) else 0.0
    return float(abs(p1 - p2))

# ---------- dataset helpers ----------
def detect_cols(name: str, df: pd.DataFrame):
    name = name.upper()
    low = {c.lower(): c for c in df.columns}
    def pick(cands):
        for c in cands:
            if c in low: return low[c]
        return None
    if name == "ADULT":
        tgt = pick(["income","label","class"]); sens = pick(["sex","gender"])
        if tgt is None:
            # Look for income-related columns (preprocessed format)
            for c in df.columns:
                if "income" in c.lower() or (df[c].astype(str).str.contains(">50").any()):
                    tgt = c
                    break
        if sens is None:
            # Look for sex-related columns (preprocessed format)
            for c in df.columns:
                if "sex" in c.lower() and ("female" in c.lower() or "male" in c.lower()):
                    sens = c
                    break
        return tgt, sens
    if name == "COMPAS":
        tgt = pick(["two_year_recid","is_recid","label"])
        sens = pick(["race","sex","gender"])
        if sens is None:
            # Look for sex or race-related columns (preprocessed format)
            for c in df.columns:
                if ("sex" in c.lower() and ("female" in c.lower() or "male" in c.lower())) or \
                   ("race" in c.lower() and any(x in c.lower() for x in ["african", "caucasian", "hispanic", "asian"])):
                    sens = c
                    break
        return tgt, sens
    if name == "GERMANCREDIT":
        tgt = pick(["credit_risk","risk","default","label"])
        sens = pick(["sex","gender","age"])
        if tgt is None:
            # Look for credit risk related columns (preprocessed format)
            for c in df.columns:
                if "credit" in c.lower() and "risk" in c.lower():
                    tgt = c
                    break
        if sens is None:
            # Look for gender-related columns (preprocessed format)
            for c in df.columns:
                if "gender" in c.lower() and "female" in c.lower():
                    sens = c
                    break
        return tgt, sens
    if name == "LAWSCHOOL":
        tgt = pick(["admit","label"])
        sens = pick(["race","race_bin","gender","sex"])
        if tgt is None:
            # Look for bar exam or admission related columns
            for c in df.columns:
                if "bar" in c.lower() or "admit" in c.lower() or "pass" in c.lower():
                    tgt = c
                    break
        if sens is None:
            # Look for gender or race related columns (preprocessed format)
            for c in df.columns:
                if ("gender" in c.lower() and "female" in c.lower()) or \
                   ("race" in c.lower() and any(x in c.lower() for x in ["black", "asian", "hisp", "other"])):
                    sens = c
                    break
        return tgt, sens
    return None, None

def make_target(name: str, df: pd.DataFrame, target_col: str) -> pd.Series:
    y = df[target_col]
    if y.dtype == object:
        return y.astype(str).str.contains(">50").astype(int)
    return (y > y.median()).astype(int) if y.nunique()>2 else y.astype(int)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

# ---------- proxy (fairness-preserving distilled set) ----------
def make_fairness_distilled_proxy(X,y,s,k_per_cell=3,random_state=0):
    df = X.copy(); df["_y_"]=y.values; df["_s_"]=s.values
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False) if cat else None
    if enc: enc.fit(X[cat])
    X_list,y_list,s_list,w_list=[],[],[],[]
    for yv in sorted(df["_y_"].unique()):
        for sv in df["_s_"].unique():
            cell = df[(df["_y_"]==yv)&(df["_s_"]==sv)]
            if cell.empty: continue
            X_cell = np.hstack([
                (cell[num].to_numpy(float) if num else np.zeros((len(cell),0))),
                (enc.transform(cell[cat]) if cat else np.zeros((len(cell),0)))
            ])
            k = min(k_per_cell, max(1,len(cell)))
            km = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit(X_cell)
            lab = km.labels_
            for ci in range(k):
                sub = cell.iloc[np.where(lab==ci)[0]]
                row={}
                if num:
                    m=sub[num].mean(0)
                    for c in num: row[c]=m[c]
                for c in cat:
                    row[c]= sub[c].mode(dropna=True).iloc[0] if not sub[c].mode(dropna=True).empty else sub[c].iloc[0]
                X_list.append(row); y_list.append(int(yv)); s_list.append(sv); w_list.append(len(sub)/len(df))
    Xp=pd.DataFrame(X_list); yp=pd.Series(y_list,name="y"); sp=pd.Series(s_list,name="s"); w=np.array(w_list,float)
    for c in X.columns:
        if c not in Xp.columns:
            Xp[c] = X[c].iloc[0] if c in cat else float(X[c].mean())
    return Xp[X.columns], yp, sp, w

# ---------- conformal fairness (shift-robust) ----------
def conformal_fairness_cert(model_pipeline, X, s, alpha=0.1, n_scenarios=40, rng=None):
    if rng is None: rng = np.random.RandomState(0)
    y_pred = model_pipeline.predict(X)
    s_arr = pd.Series(s).values.copy(); uniq = pd.unique(s_arr)
    if len(uniq) < 2: return {"q_hat":0.0,"samples":[0.0]}
    vals=[]
    for _ in range(n_scenarios):
        probs = rng.dirichlet(np.ones(len(uniq)))
        idxs=[]
        for gi,g in enumerate(uniq):
            gidx = np.where(s_arr==g)[0]
            k = max(1, int(len(gidx)*probs[gi]))
            idxs.append(rng.choice(gidx, size=k, replace=True))
        idxs=np.concatenate(idxs)
        flip = rng.rand(len(idxs)) < 0.05
        s_cf = s_arr[idxs].copy()
        if len(uniq)==2:
            s_cf[flip] = np.where(s_cf[flip]==uniq[0], uniq[1], uniq[0])
        vals.append(differential_statistical_parity(y_pred[idxs], s_cf))
    return {"q_hat": float(np.quantile(vals, 1-alpha)), "samples": vals}

# ---------- energy estimator (Green Algorithms approx) ----------
def green_algorithms_energy(runtime_sec, cpu_cores=4, mem_gb=8.0, pue=1.58, ci_g_per_kwh=475.0):
    t_h = max(runtime_sec,0.01)/3600.0
    P_core, P_mem = 10.8, 0.3725
    kWh = t_h * (cpu_cores*P_core + mem_gb*P_mem) * pue / 1000.0
    return {"kWh": float(kWh), "gCO2e": float(kWh*ci_g_per_kwh)}

# ---------- config ----------
@dataclass
class Config:
    dataset_name: str
    dataset_path: Path
    test_size: float = 0.2
    random_state: int = 42
    rounds: int = 20
    proxy_every: int = 1
    alpha: float = 0.1       # conformal level
    tau: float = 0.2         # fairness threshold on DSP
    k_per_cell: int = 3
    cpu_cores: int = 4
    mem_gb: float = 8.0
    pue: float = 1.58
    ci_g_per_kwh: float = 475.0

# ---------- optimizer (model-agnostic) ----------
class CafeMOOptimizer:
    """
    model_fn: Callable[[Dict[str,Any]], sklearn-like estimator] -> returns model (e.g., RF, LogReg, ...)
    sample_params_fn: Callable[[np.random.RandomState], Dict[str,Any]] -> samples a hyperparameter dict
    """
    def __init__(self, cfg: Config, model_fn: Callable[[Dict[str,Any]], Any],
                 sample_params_fn: Callable[[np.random.RandomState], Dict[str,Any]]):
        self.cfg = cfg
        self.model_fn = model_fn
        self.sample_params_fn = sample_params_fn
        self.rng = np.random.RandomState(cfg.random_state)
        self.logs: List[Dict[str,Any]] = []
        self._prep_data()
        self.pre = build_preprocessor(self.X_train)
        self.surr_err = RandomForestRegressor(n_estimators=200, random_state=0)
        self.surr_dsp = RandomForestRegressor(n_estimators=200, random_state=0)
        self.surr_energy = RandomForestRegressor(n_estimators=200, random_state=0)
        self.surr_fitted = False
        self.Xp, self.yp, self.sp, self.wp = make_fairness_distilled_proxy(
            self.X_train, self.y_train, self.s_train, k_per_cell=self.cfg.k_per_cell, random_state=self.cfg.random_state
        )

    def _prep_data(self):
        df = pd.read_csv(self.cfg.dataset_path)
        ycol, scol = detect_cols(self.cfg.dataset_name, df)
        if not ycol or not scol: raise ValueError("could not detect target/sensitive columns")
        y = make_target(self.cfg.dataset_name, df, ycol); s=df[scol]; X=df.drop(columns=[ycol])
        self.X_train,self.X_test,self.y_train,self.y_test,self.s_train,self.s_test = train_test_split(
            X, y, s, test_size=self.cfg.test_size, random_state=self.cfg.random_state, stratify=y)

    def _train_eval(self, params: Dict[str,Any], use_proxy: bool):
        t0 = time.time()
        model = self.model_fn(params)
        pipe = Pipeline([("pre", self.pre), ("clf", model)])
        if use_proxy:
            pipe.fit(self.Xp, self.yp, clf__sample_weight=self.wp); y_pred = pipe.predict(self.X_test)
        else:
            pipe.fit(self.X_train, self.y_train);                  y_pred = pipe.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        dsp = differential_statistical_parity(y_pred, self.s_test)
        runtime = time.time()-t0
        cert = conformal_fairness_cert(pipe, self.X_test, self.s_test, alpha=self.cfg.alpha, n_scenarios=40, rng=self.rng)
        E = green_algorithms_energy(runtime, self.cfg.cpu_cores, self.cfg.mem_gb, self.cfg.pue, self.cfg.ci_g_per_kwh)
        return {"params": params, "accuracy": float(acc), "dsp": float(dsp),
                "kWh": E["kWh"], "gCO2e": E["gCO2e"], "kind": "proxy" if use_proxy else "full",
                "time_sec": float(runtime), "q_hat": cert["q_hat"]}

    def _update_surrogates(self):
        if len(self.logs) < 3: return
        rows = []
        for r in self.logs:
            flat = {k: (float(v) if isinstance(v,(int,float)) else 0.0) for k,v in r["params"].items()}
            flat.update({"err": 1-r["accuracy"], "dsp": r["dsp"], "energy": r["kWh"]})
            rows.append(flat)
        df = pd.DataFrame(rows).fillna(0.0)
        Xs = df.drop(columns=["err","dsp","energy"]).to_numpy()
        self.surr_err.fit(Xs, df["err"]); self.surr_dsp.fit(Xs, df["dsp"]); self.surr_energy.fit(Xs, df["energy"])
        self.surr_fitted = True

    def _score_by_surrogate(self, params: Dict[str,Any]) -> float:
        if not self.surr_fitted: return 0.0
        x = pd.DataFrame([{k:(float(v) if isinstance(v,(int,float)) else 0.0) for k,v in params.items()}]).to_numpy()
        err = self.surr_err.predict(x)[0]; dsp = self.surr_dsp.predict(x)[0]; ene = self.surr_energy.predict(x)[0]
        w = np.random.default_rng().dirichlet([1,1,1])
        return - (w[0]*err + w[1]*dsp + w[2]*ene)

    def run(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir/"cafe_mo_log.jsonl"
        for t in range(self.cfg.rounds):
            last_result = None
            # propose by surrogate
            pool = [self.sample_params_fn(self.rng) for _ in range(8)]
            cand = max(pool, key=self._score_by_surrogate)
            res_p = self._train_eval(cand, use_proxy=True)
            last_result = res_p
            if res_p["q_hat"] <= self.cfg.tau:
                self.logs.append(res_p); self._update_surrogates()
            # full eval periodically
            if t % (self.cfg.proxy_every + 1) == self.cfg.proxy_every:
                pool = [self.sample_params_fn(self.rng) for _ in range(6)]
                if self.surr_fitted: pool.sort(key=self._score_by_surrogate, reverse=True)
                res_f = self._train_eval(pool[0], use_proxy=False)
                last_result = res_f
                self.logs.append(res_f); self._update_surrogates()
            # Always log something
            log_entry = self.logs[-1] if self.logs else last_result
            with open(log_path,"a",encoding="utf-8") as f:
                f.write(json.dumps({"t":t,"last":log_entry}, default=float)+"\n")
        # outputs
        if self.logs:
            df = pd.DataFrame(self.logs); df["err"] = 1-df["accuracy"]
            df.to_csv(out_dir/"cafe_mo_all.csv", index=False)
            pareto = self._pareto(df); pareto.to_csv(out_dir/"cafe_mo_pareto.csv", index=False)
            return df, pareto
        else:
            # Create minimal dataframe from log file if no results passed fairness constraint
            log_data = []
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        log_data.append(entry["last"])
            df = pd.DataFrame(log_data) if log_data else pd.DataFrame()
            if not df.empty:
                df["err"] = 1-df["accuracy"]
                df.to_csv(out_dir/"cafe_mo_all.csv", index=False)
                pareto = self._pareto(df); pareto.to_csv(out_dir/"cafe_mo_pareto.csv", index=False)
                return df, pareto
            else:
                # Return empty dataframes
                empty_df = pd.DataFrame()
                return empty_df, empty_df

    @staticmethod
    def _pareto(df: pd.DataFrame) -> pd.DataFrame:
        vals = df[["err","dsp","kWh"]].to_numpy()
        nd = np.ones(len(vals), bool)
        for i in range(len(vals)):
            if not nd[i]: continue
            dominated = (np.all(vals <= vals[i], 1) & np.any(vals < vals[i], 1))
            nd[dominated] = False
        return df[nd].sort_values(["err","dsp","kWh"]).reset_index(drop=True)
