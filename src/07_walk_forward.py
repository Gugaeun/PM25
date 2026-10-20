"""Walk-forward(확장 윈도우) 평가: 단일 분할의 한계(테스트가 여름·가을 위주) 확인.

fold k: 테스트 = 2025년 분기 k (1Q~4Q), 학습 = 테스트 이전 전부(2024-01 ~),
        그중 테스트 직전 60일은 조기 종료용 검증으로 분리. 모든 fold 에서 같은 모델/하이퍼파라미터.
모델: naive, persistence, LightGBM, LSTM, 1D-CNN (PM2.5만 / +기상 변수). NN 은 시드 3개 평균.
산출: results/walkforward.csv, results/walkforward_summary.csv, results/figures/walkforward.png
실행: python src/07_walk_forward.py [--seeds 3]
"""
import argparse
import importlib.util
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_split import HORIZONS, MIN_HISTORY, ROOT, load, metrics

torch.set_num_threads(8)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bl = _load("bl", "02_baselines.py")
pt = _load("pt", "04_pytorch.py")
OUT = ROOT / "results" / "walkforward"
OUT.mkdir(parents=True, exist_ok=True)

FOLDS = [("2025Q1", "2025-01-01", "2025-03-31 23:00"), ("2025Q2", "2025-04-01", "2025-06-30 23:00"),
         ("2025Q3", "2025-07-01", "2025-09-30 23:00"), ("2025Q4", "2025-10-01", "2025-12-31 23:00")]
VAL_DAYS = 60

LABELS = {"lgbm_base": "LightGBM", "lgbm_weather": "LightGBM\n+기상", "lstm_base": "LSTM", "lstm_weather": "LSTM\n+기상",
          "cnn_base": "1D-CNN", "cnn_weather": "1D-CNN\n+기상"}


def plot_walkforward(res, path):
    """분기별 persistence 대비 MAE 감소율. 음수 = persistence 보다 나쁨."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    pv = res.pivot(index="model", columns="fold", values="MAE_red_vs_persistence_%").loc[list(LABELS)]
    colors = ["#9fb8e6", "#6f97dc", "#3f73c6", "#1b4a96"]
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.2
    x = np.arange(len(pv))
    for j, f in enumerate(pv.columns):
        ax.bar(x + (j - 1.5) * w, pv[f].values, width=w, color=colors[j], label=f.replace("2025", "") + " 분기")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in pv.index])
    ax.set_ylabel("persistence 대비 MAE 감소율 (%)")
    ax.set_title("Walk-forward: 분기별 개선율 (음수 = persistence보다 나쁨)")
    ax.grid(alpha=.3, axis="y")
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fold_origins(df, a, b):
    idx = df.index
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    vs = a - pd.Timedelta(days=VAL_DAYS)

    def sel(lo, hi):  # origin 이 [lo, hi] 안이고 타깃(t+24)도 hi 를 넘지 않음
        pos = np.where((idx >= lo) & (idx <= hi))[0]
        pos = pos[pos + 24 <= np.where(idx <= hi)[0].max()]
        return pos[pos >= MIN_HISTORY]

    return {"train": sel(idx[0], vs - pd.Timedelta(hours=1)), "val": sel(vs, a - pd.Timedelta(hours=1)), "test": sel(a, b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    df = load()
    y = df["PM25"].values
    rows = []
    for fold, a, b in FOLDS:
        S = fold_origins(df, a, b)
        Yt = np.stack([y[S["test"] + h] for h in HORIZONS], 1)
        print(f"== {fold}: train={len(S['train'])} val={len(S['val'])} test={len(S['test'])} "
              f"(test 평균 PM2.5 {Yt.mean():.1f}, 75초과 {int((Yt > 75).sum())}개)", flush=True)
        preds = {"naive": bl.naive_pred(y, S["test"]), "persistence": bl.persistence_pred(y, S["test"])}
        for g in ("base", "weather"):
            f = OUT / f"{fold}_lgbm_{g}.npy"
            if f.exists():
                preds[f"lgbm_{g}"] = np.load(f)
                continue
            wc = [c for c in pt.feature_cols(g) if c != "PM25"]
            preds[f"lgbm_{g}"] = bl.lgbm_pred(df, S, wc)["test"]
            np.save(f, preds[f"lgbm_{g}"])
        for m in ("lstm", "cnn"):
            for g in ("base", "weather"):
                f = OUT / f"{fold}_{m}_{g}.npy"
                if f.exists():
                    preds[f"{m}_{g}"] = np.load(f)
                    continue
                t0 = time.time()
                runs = [pt.run_one(m, g, 48, s, df, S)[0]["test"] for s in range(args.seeds)]
                np.save(OUT / f"{fold}_{m}_{g}_seeds.npy", np.stack(runs))
                preds[f"{m}_{g}"] = np.mean(runs, 0)  # 시드 앙상블
                np.save(f, preds[f"{m}_{g}"])
                print(f"   {m}_{g} {time.time() - t0:.0f}s", flush=True)
        d23 = df.index[S["test"]].hour == 23
        for k, p in preds.items():
            rows.append({"fold": fold, "model": k, **metrics(Yt, p),
                         "h24_MAE": float(np.abs(p[:, 23] - Yt[:, 23]).mean()),
                         "daily_MAE": float(np.abs(p[d23].mean(1) - Yt[d23].mean(1)).mean()),
                         "mean_PM25": float(Yt.mean()), "n_gt75": int((Yt > 75).sum())})
        print(pd.DataFrame([r for r in rows if r["fold"] == fold])[["model", "MAE", "RMSE"]].round(2).to_string(index=False), flush=True)

    res = pd.DataFrame(rows)
    res["MAE_red_vs_persistence_%"] = res.apply(
        lambda r: 100 * (1 - r.MAE / res[(res.fold == r.fold) & (res.model == "persistence")].MAE.iloc[0]), axis=1)
    res.to_csv(ROOT / "results" / "walkforward.csv", index=False)
    summ = res.groupby("model").agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), red_mean=("MAE_red_vs_persistence_%", "mean"),
                                    red_min=("MAE_red_vs_persistence_%", "min"), red_max=("MAE_red_vs_persistence_%", "max"))
    summ.round(3).to_csv(ROOT / "results" / "walkforward_summary.csv")
    print(summ.round(2).to_string())

    plot_walkforward(res, ROOT / "results" / "figures" / "walkforward.png")


if __name__ == "__main__":
    main()
