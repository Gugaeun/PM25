"""하이퍼파라미터 튜닝 확인: 기준 모델이 약해서 딥러닝이 이긴 것은 아닌가.

기본 설정으로 학습한 LightGBM 과 LSTM/CNN 을, 검증 구간(2025-01~06)에서 설정을 찾은 버전과 비교한다.
설정은 검증 구간 MAE 로만 고르고, 테스트는 고른 뒤 한 번만 본다.
  - LightGBM: 무작위 탐색 16개 설정 (num_leaves, learning_rate, min_child_samples, subsample, colsample, reg_lambda)
  - CNN/LSTM(기상 변수 포함): 은닉 크기·dropout·학습률·weight decay 조합 6개, 시드 1개로 고른 뒤 최선만 시드 3개로 재학습
결과 해석: 튜닝해도 순위(persistence 대비 개선, 기상 변수 효과)가 그대로면 결론은 설정 선택의 영향이 작다는 뜻이다.

실행: python src/13_tuning_check.py   →  results/tuning_check.csv, results/tuning_check.md
"""
import importlib.util
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from data_split import HORIZONS, ROOT, SEED, load, split_origins


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bl = _load("bl", "02_baselines.py")
pt = _load("pt", "04_pytorch.py")
OUT = ROOT / "results"


def mae(P, Y):
    return float(np.abs(P - Y).mean())


# ---------------- LightGBM ----------------
def lgbm_search(df, S, y, group, n_cfg=16):
    wc = [c for c in pt.feature_cols(group) if c != "PM25"]
    Xs = {k: bl.make_features(df, S[k], wc) for k in ("train", "val", "test")}
    Ys = {k: np.stack([y[S[k] + h] for h in HORIZONS], 1) for k in ("train", "val", "test")}
    rng = np.random.RandomState(SEED)
    cfgs = [dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0)]  # 기본(비교 기준)
    for _ in range(n_cfg):
        cfgs.append(dict(n_estimators=1000, learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])), num_leaves=int(rng.choice([7, 15, 31, 63])),
                         min_child_samples=int(rng.choice([20, 40, 80, 160])), subsample=float(rng.choice([0.6, 0.8, 1.0])),
                         colsample_bytree=float(rng.choice([0.5, 0.7, 0.9])), reg_lambda=float(rng.choice([0.0, 1.0, 10.0]))))

    def fit_all(cfg, hs):
        P = {k: np.zeros((len(Xs[k]), len(HORIZONS))) for k in ("val", "test")}
        for h in hs:
            m = lgb.LGBMRegressor(subsample_freq=1, random_state=SEED, verbose=-1, **cfg)
            m.fit(Xs["train"], Ys["train"][:, h - 1], eval_set=[(Xs["val"], Ys["val"][:, h - 1])], callbacks=[lgb.early_stopping(50, verbose=False)])
            for k in P:
                P[k][:, h - 1] = m.predict(Xs[k])
        return P

    quick = (1, 6, 12, 24)  # 탐색은 4개 horizon 만으로 빠르게
    scores = []
    for i, cfg in enumerate(cfgs):
        P = fit_all(cfg, quick)
        scores.append(float(np.mean([np.abs(P["val"][:, h - 1] - Ys["val"][:, h - 1]).mean() for h in quick])))
        print(f"  lgbm[{group}] cfg {i}: val MAE(1·6·12·24h) {scores[i]:.3f}", flush=True)
    best = int(np.argmin(scores))
    Pd = fit_all(cfgs[0], HORIZONS)
    Pb = fit_all(cfgs[best], HORIZONS) if best else Pd
    return {"default": (mae(Pd["val"], Ys["val"]), mae(Pd["test"], Ys["test"])), "tuned": (mae(Pb["val"], Ys["val"]), mae(Pb["test"], Ys["test"])),
            "best_cfg": cfgs[best], "is_default_best": best == 0}


# ---------------- 신경망 ----------------
def run_nn(model, cfg, df, S, seed):
    """04_pytorch.run_one 을 크기·dropout·학습률·decay 만 바꿔 실행(원본 코드는 그대로)."""
    orig = (pt.LSTMNet.__init__.__defaults__, pt.CNNNet.__init__.__defaults__, torch.optim.AdamW)
    pt.LSTMNet.__init__.__defaults__ = (cfg["hidden"], 2, cfg["drop"])
    pt.CNNNet.__init__.__defaults__ = (cfg["hidden"], cfg["drop"])
    lr, wd = cfg["lr"], cfg["wd"]

    class Opt(torch.optim.AdamW):
        def __init__(self, params, **kw):
            kw["lr"], kw["weight_decay"] = lr, wd  # run_one 이 넘기는 값 대신 탐색 중인 값을 쓴다
            super().__init__(params, **kw)
    pt.torch.optim.AdamW = Opt
    try:
        P, _ = pt.run_one(model, "weather", 48, seed, df, S)
    finally:
        pt.LSTMNet.__init__.__defaults__, pt.CNNNet.__init__.__defaults__ = orig[0], orig[1]
        pt.torch.optim.AdamW = orig[2]
    return P


def nn_search(model, df, S, y):
    Y = {k: np.stack([y[S[k] + h] for h in HORIZONS], 1) for k in ("val", "test")}
    base = dict(hidden=64, drop=0.1, lr=2e-3, wd=1e-4)
    cfgs = [base, dict(hidden=32, drop=0.1, lr=2e-3, wd=1e-4), dict(hidden=128, drop=0.1, lr=2e-3, wd=1e-4), dict(hidden=64, drop=0.3, lr=2e-3, wd=1e-4),
            dict(hidden=64, drop=0.1, lr=1e-3, wd=1e-4), dict(hidden=128, drop=0.3, lr=1e-3, wd=1e-3)]
    scores = []
    for i, c in enumerate(cfgs):
        P = run_nn(model, c, df, S, 0)
        scores.append(mae(P["val"], Y["val"]))
        print(f"  {model} cfg {i} {c}: val MAE {scores[i]:.3f}", flush=True)
    best = int(np.argmin(scores))
    res = {"default_cfg": base, "best_cfg": cfgs[best], "is_default_best": best == 0}
    for name, c in (("default", base), ("tuned", cfgs[best])):
        runs = [run_nn(model, c, df, S, sd)for sd in range(3)] if (name == "default" or best) else None
        if runs is None:
            res[name] = res["default"]
            continue
        ens = {k: np.mean([r[k] for r in runs], 0) for k in ("val", "test")}
        res[name] = (mae(ens["val"], Y["val"]), mae(ens["test"], Y["test"]))
    return res


def main():
    t0 = time.time()
    df = load()
    S = split_origins(df)
    y = df["PM25"].values
    rows = []
    for group in ("base", "weather"):
        r = lgbm_search(df, S, y, group)
        for k in ("default", "tuned"):
            rows.append({"모델": f"LightGBM({'PM2.5만' if group == 'base' else '+기상'})", "설정": k, "검증 MAE": r[k][0], "테스트 MAE": r[k][1], "선택된 설정": str(r["best_cfg"]) if k == "tuned" else "기본", "기본이 최선": r["is_default_best"]})
    for model in ("cnn", "lstm"):
        r = nn_search(model, df, S, y)
        for k in ("default", "tuned"):
            rows.append({"모델": f"{'1D-CNN' if model == 'cnn' else 'LSTM'}+기상", "설정": k, "검증 MAE": r[k][0], "테스트 MAE": r[k][1], "선택된 설정": str(r["best_cfg"]) if k == "tuned" else str(r["default_cfg"]), "기본이 최선": r["is_default_best"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tuning_check.csv", index=False, encoding="utf-8-sig")
    md = ["# 튜닝 확인 (설정은 검증 구간으로 선택, 테스트는 한 번)", "", "| 모델 | 설정 | 검증 MAE | 테스트 MAE | 선택된 설정 |", "|---|---|---|---|---|"]
    md += [f"| {r['모델']} | {r['설정']} | {r['검증 MAE']:.3f} | {r['테스트 MAE']:.3f} | {r['선택된 설정']} |" for _, r in out.iterrows()]
    (OUT / "tuning_check.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(out[["모델", "설정", "검증 MAE", "테스트 MAE", "기본이 최선"]].round(3).to_string(index=False))
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
