"""결과 분석: horizon별 오차 그래프, 실제 vs 예측, ablation 표, 고농도 구간 실패 분석.

입력: results/baselines_*, results/pytorch/*  (02, 04 실행 후)
출력: results/figures/*.png, results/ablation_table.md, results/failure_analysis.md
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_split import HORIZONS, ROOT, load, split_origins

warnings.filterwarnings("ignore")
RES = ROOT / "results"
PT = RES / "pytorch"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    try:
        plt.rcParams["font.family"] = f
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

df = load()
S = split_origins(df)
y = df["PM25"].values
pos = S["test"]
Y = np.stack([y[pos + h] for h in HORIZONS], 1)


def mae_h(P, idx=slice(None)):
    return np.abs(P[idx] - Y[idx]).mean(0)


# ---------- 예측값 모음 (test) ----------
preds = {
    "naive (어제 같은 시각)": np.load(RES / "baselines_test_pred_naive_yesterday.npy"),
    "persistence": np.load(RES / "baselines_test_pred_persistence.npy"),
    "LightGBM": np.load(RES / "baselines_test_pred_lightgbm.npy"),
}
for label, name in (("LSTM", "lstm_base_L48"), ("1D-CNN", "cnn_base_L48")):
    f = PT / f"{name}_test_ens_pred.npy"
    if f.exists():
        preds[label] = np.load(f)
for label, name in (("LSTM+기상", "lstm_weather_L48"), ("LightGBM+기상", "lgbm_weather_L48")):
    f = PT / f"{name}_test_ens_pred.npy"
    if f.exists():
        preds[label] = np.load(f)

# ---------- 1. horizon별 MAE ----------
fig, ax = plt.subplots(figsize=(8, 5))
for k, P in preds.items():
    ax.plot(HORIZONS, mae_h(P), marker="o", ms=3, label=k, ls="--" if "naive" in k else "-")
ax.set_xlabel("예측 horizon (시간)")
ax.set_ylabel("MAE (µg/m³)")
ax.set_title("테스트(2025-07~12) horizon별 MAE")
ax.grid(alpha=.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "horizon_mae.png", dpi=150)
plt.close(fig)

# ---------- 2. 실제 vs 예측 ----------
best = "LSTM+기상" if "LSTM+기상" in preds else ("LSTM" if "LSTM" in preds else "LightGBM")
ts = df.index[pos]
for h in (1, 6, 24):
    # 테스트 중 가장 PM2.5 가 높았던 2주 + 평균적인 2주
    s = pd.Series(Y[:, h - 1], index=ts + pd.Timedelta(hours=h))
    peak_start = s.rolling(24 * 14).mean().idxmax() - pd.Timedelta(days=13)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, (t0, ttl) in zip(axes, ((peak_start, "고농도 2주"), (pd.Timestamp("2025-09-01"), "평균적 2주 (9월)"))):
        m = (s.index >= t0) & (s.index < t0 + pd.Timedelta(days=14))
        ax.plot(s.index[m], s[m], "k", lw=1.2, label="실제")
        ax.plot(s.index[m], preds[best][m, h - 1], "r", lw=1, label=f"{best} 예측")
        ax.plot(s.index[m], preds["persistence"][m, h - 1], color="gray", lw=.8, alpha=.7, label="persistence")
        ax.set_title(f"{h}시간 뒤 예측 — {ttl}")
        ax.set_ylabel("PM2.5")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / f"actual_vs_pred_h{h}.png", dpi=150)
    plt.close(fig)

# ---------- 3. ablation 표 + 그림 ----------
rows = []
for f in PT.glob("*_L48.json"):
    rows += json.loads(f.read_text())
if rows:
    ab = pd.DataFrame(rows)
    ab = ab[(ab.split == "test") & ab.model.isin(["lstm", "lgbm"]) & (ab.window == 48)]
    pv = ab.pivot(index="group", columns="model", values="MAE")
    sd = ab.pivot(index="group", columns="model", values="MAE_std")
    order = [g for g in ["base", "temp_humid", "ws", "wind_uv", "weather", "pollutants", "all"] if g in pv.index]
    lines = ["| 입력 변수 | LSTM MAE (±std, 3시드) | LSTM 개선(vs PM2.5만) | LightGBM MAE | LightGBM 개선 |", "|---|---|---|---|---|"]
    for g in order:
        r = pv.loc[g]
        b = pv.loc["base"]
        def fmt(m):
            if m not in pv.columns or np.isnan(r.get(m, np.nan)):
                return "–", "–"
            std = f" ±{sd.loc[g, m]:.2f}" if sd.loc[g, m] > 0 else ""
            return f"{r[m]:.3f}{std}", f"{100 * (1 - r[m] / b[m]):+.1f}%"
        l, lg = fmt("lstm"), fmt("lgbm")
        lines.append(f"| {g} | {l[0]} | {l[1]} | {lg[0]} | {lg[1]} |")
    (RES / "ablation_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # horizon별 개선 (LSTM, base 대비)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, ls in (("lstm", "-"), ("lgbm", "--")):
        b = pd.read_csv(PT / f"{model}_base_L48_test_horizon.csv", index_col=0)["MAE"]
        for g in ("weather", "pollutants", "all"):
            f = PT / f"{model}_{g}_L48_test_horizon.csv"
            if f.exists():
                m = pd.read_csv(f, index_col=0)["MAE"]
                ax.plot(HORIZONS, 100 * (1 - m / b), ls=ls, label=f"{model} + {g}")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("예측 horizon (시간)")
    ax.set_ylabel("PM2.5만 쓴 모델 대비 MAE 개선 (%)")
    ax.set_title("변수 추가 효과: horizon별")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_by_horizon.png", dpi=150)
    plt.close(fig)

# ---------- 4. 고농도 구간 실패 분석 ----------
P = preds[best]
lv = pd.cut(Y.ravel(), [-1, 15, 35, 75, 1e9], labels=["좋음(≤15)", "보통(16~35)", "나쁨(36~75)", "매우나쁨(>75)"])
rep = []
for h in (1, 6, 24):
    for k, Pm in (("persistence", preds["persistence"]), (best, P)):
        e = Pm[:, h - 1] - Y[:, h - 1]
        d = pd.DataFrame({"lv": pd.cut(Y[:, h - 1], [-1, 15, 35, 75, 1e9], labels=["좋음(≤15)", "보통(16~35)", "나쁨(36~75)", "매우나쁨(>75)"]), "e": e})
        g = d.groupby("lv", observed=True).e.agg(n="size", MAE=lambda x: x.abs().mean(), bias="mean")
        g["h"], g["model"] = h, k
        rep.append(g.reset_index())
rep = pd.concat(rep)
out = ["# 고농도 구간 실패 분석 (테스트)", "", f"모델: {best}. bias = 평균(예측−실제), 음수면 과소예측.", "",
       rep.round(2).to_markdown(index=False) if hasattr(rep, "to_markdown") else rep.round(2).to_string(index=False)]
# 최악 사건: 24h 뒤 예측에서 오차 큰 날
e24 = pd.Series(np.abs(P[:, 23] - Y[:, 23]), index=ts + pd.Timedelta(hours=24))
worst = e24.resample("D").mean().sort_values(ascending=False).head(5)
out += ["", "## 24h 뒤 예측 일평균 절대오차 상위 5일", "", worst.round(1).to_string()]
(RES / "failure_analysis.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))

fig, ax = plt.subplots(figsize=(6, 6))
h = 24
ax.scatter(Y[:, h - 1], P[:, h - 1], s=4, alpha=.3)
mx = max(Y[:, h - 1].max(), P[:, h - 1].max())
ax.plot([0, mx], [0, mx], "r--", lw=1)
ax.set_xlabel("실제 PM2.5 (24h 뒤)")
ax.set_ylabel(f"{best} 예측")
ax.set_title("고농도에서 과소예측 경향")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "scatter_h24.png", dpi=150)
plt.close(fig)
print("done; figures in", FIG)
