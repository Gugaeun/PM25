"""results/ 의 산출물을 웹 리포트용 JSON 으로 모은다. 실행(루트): python report/build_report_data.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_split import HORIZONS, load, split_origins

R, PT = ROOT / "results", ROOT / "results" / "pytorch"
df = load(); S = split_origins(df); pos = S["test"]; y = df["PM25"].values
Y = np.stack([y[pos + h] for h in HORIZONS], 1)
P = {
 "naive": np.load(R/"baselines_test_pred_naive_yesterday.npy"), "persistence": np.load(R/"baselines_test_pred_persistence.npy"),
 "lgbm": np.load(R/"baselines_test_pred_lightgbm.npy"), "lstm": np.load(PT/"lstm_base_L48_test_ens_pred.npy"),
 "cnn": np.load(PT/"cnn_base_L48_test_ens_pred.npy"), "lstm_w": np.load(PT/"lstm_weather_L48_test_ens_pred.npy"),
 "cnn_w": np.load(PT/"cnn_weather_L48_test_ens_pred.npy"), "lgbm_w": np.load(PT/"lgbm_weather_L48_test_ens_pred.npy")}
d = {}
d["horizon_mae"] = {k: np.abs(v - Y).mean(0).round(3).tolist() for k, v in P.items()}
d["overall"] = {k: round(float(np.abs(v - Y).mean()), 3) for k, v in P.items()}
m23 = df.index[pos].hour == 23
d["daily"] = {k: round(float(np.abs(v[m23].mean(1) - Y[m23].mean(1)).mean()), 3) for k, v in P.items()}
rows = []
for f in PT.glob("*_L48.json"): rows += json.loads(f.read_text())
ab = pd.DataFrame(rows); ab = ab[(ab.split == "test") & ab.model.isin(["lstm", "lgbm"])]
d["ablation"] = {m: {r.group: [round(r.MAE, 3), round(r.MAE_std, 3)] for r in ab[ab.model == m].itertuples()} for m in ("lstm", "lgbm")}
wd = pd.DataFrame([x for f in PT.glob("lstm_base_L*.json") for x in json.loads(f.read_text())])
wd = wd[wd.split == "test"].sort_values("window"); d["window"] = {int(r.window): round(r.MAE, 3) for r in wd.itertuples()}
# 실제 vs 예측 구간
ts = df.index[pos]
d["series"] = {}
for h in (1, 6, 24):
    s = pd.Series(Y[:, h-1], index=ts + pd.Timedelta(hours=h))
    peak = s.rolling(24*14).mean().idxmax() - pd.Timedelta(days=13)
    out = {}
    for name, t0 in (("peak", peak), ("calm", pd.Timestamp("2025-09-01"))):
        m = (s.index >= t0) & (s.index < t0 + pd.Timedelta(days=14))
        out[name] = {"start": t0.strftime("%Y-%m-%d"), "actual": s[m].round(1).tolist(),
                     "lstm_w": P["lstm_w"][m, h-1].round(1).tolist(), "persistence": P["persistence"][m, h-1].round(1).tolist()}
    d["series"][h] = out
# 농도 구간별 편향
bins = [-1, 15, 35, 75, 1e9]; lab = ["좋음", "보통", "나쁨", "매우나쁨"]; fa = []
for h in (1, 6, 24):
    for k in ("persistence", "lstm_w"):
        e = P[k][:, h-1] - Y[:, h-1]; lv = pd.cut(Y[:, h-1], bins, labels=lab)
        for l in lab:
            m = lv == l; fa.append({"h": h, "model": k, "level": l, "n": int(m.sum()), "mae": round(float(np.abs(e[m]).mean()), 2), "bias": round(float(e[m].mean()), 2)})
d["failure"] = fa

# 고농도 탐지(나쁨 이상, 기본 임계값)와 손실 함수 실험
ev = pd.read_csv(R/"event_metrics.csv", encoding="utf-8-sig")
x = ev[(ev["기준"] == "나쁨 이상 (>35)") & ev["예측"].isin(["6h 뒤", "24h 뒤"])]
order = ["persistence", "LightGBM", "LSTM", "1D-CNN", "LSTM+기상", "1D-CNN+기상"]
def cell(m, h, col):
    v = x[(x["모델"] == m) & (x["예측"] == h)][col].iloc[0]
    return None if pd.isna(v) else round(float(v), 3)
d["events"] = {"n_events": int(x["테스트 고농도 수"].iloc[0]), "n_total": int(x["전체 수"].iloc[0]),
               "rows": [{"model": m, "r6": cell(m, "6h 뒤", "기본 재현율"), "p6": cell(m, "6h 뒤", "기본 정밀도"), "r24": cell(m, "24h 뒤", "기본 재현율"),
                         "p24": cell(m, "24h 뒤", "기본 정밀도"), "ap24": cell(m, "24h 뒤", "AP")} for m in order]}
lo = pd.read_csv(R/"event_loss.csv", encoding="utf-8-sig")
pers24 = [r for r in fa if r["h"] == 24 and r["model"] == "persistence" and r["level"] == "나쁨"][0]["bias"]
d["loss"] = {"persistence_bias24": pers24,
             "rows": [{"model": r["모델"], "loss": r["손실"], "mae": round(float(r["MAE(1~24h)"]), 2), "bias24": round(float(r["24h 편향(나쁨)"]), 1),
                       "r6": round(float(r["6h 재현율"]), 2), "p6": round(float(r["6h 정밀도"]), 2), "r24": round(float(r["24h 재현율"]), 2),
                       "p24": round(float(r["24h 정밀도"]), 2), "ap24": round(float(r["24h AP"]), 2)} for _, r in lo.iterrows()]}

# 장기 데이터(2026 새 기간, 학습 3년) 결과
lg = pd.read_csv(R/"longer_data.csv", encoding="utf-8-sig")
models = ["1D-CNN+기상", "LSTM+기상", "LightGBM+기상", "LightGBM(PM2.5만)"]
cols = [("2024 (1년)", "2025-07~12"), ("2024 (1년)", "2026-01~09"), ("2022~2024 (3년)", "2025-07~12"), ("2022~2024 (3년)", "2026-01~09")]
def pick(tr, ev, m):
    return lg[(lg["학습"] == tr) & (lg["평가 구간"] == ev) & (lg["모델"] == m)].iloc[0]
d["long"] = {"cols": [f"학습 {'1년' if tr.startswith('2024') else '3년'} · {ev}" for tr, ev in cols],
             "pers": [round(float(pick(tr, ev, "persistence")["MAE"]), 2) for tr, ev in cols],
             "n_high_2026": int(pick(cols[1][0], cols[1][1], "persistence")["고농도 시간(>35)"]),
             "rows": [{"model": m, "cells": [[round(float(pick(tr, ev, m)["persistence 대비 %"]), 1), round(float(pick(tr, ev, m)["구간 하한"]), 1), round(float(pick(tr, ev, m)["구간 상한"]), 1)] for tr, ev in cols]} for m in models]}
# 학습 곡선(GPU, 시드 5개): 학습 기간별 persistence 대비 감소율, 평가 2026-01~09
lc = pd.read_csv(R/"learning_curve.csv", encoding="utf-8-sig")
sets = list(dict.fromkeys(lc["학습"]))
lc26 = lc[lc["평가 구간"] == "2026-01~09"]
def lcv(s, m, col):
    return float(lc26[(lc26["학습"] == s) & (lc26["모델"] == m)].iloc[0][col])
d["curve"] = {"sets": sets, "pers": round(lcv(sets[0], "persistence", "MAE"), 2), "pers_bias24": round(lcv(sets[0], "persistence", "24h 편향(나쁨)"), 1),
              "rows": [{"model": m, "red": [round(lcv(s, m, "persistence 대비 %"), 1) for s in sets], "mae": [round(lcv(s, m, "MAE"), 2) for s in sets]} for m in models],
              "bias24": [round(lcv(s, "1D-CNN+기상", "24h 편향(나쁨)"), 1) for s in sets]}
wf = pd.read_csv(R/"walkforward.csv")
wf["model"] = wf.model.replace({"lstm_base":"lstm","lstm_weather":"lstm_w","cnn_base":"cnn","cnn_weather":"cnn_w","lgbm_base":"lgbm","lgbm_weather":"lgbm_w"})
d["wf"] = {"mae": {k: g.set_index("fold").MAE.round(3).to_dict() for k, g in wf.groupby("model")},
           "red": {k: g.set_index("fold")["MAE_red_vs_persistence_%"].round(1).to_dict() for k, g in wf.groupby("model")},
           "mean_pm25": wf.groupby("fold").mean_PM25.first().round(1).to_dict(),
           "n_hours_gt75": (wf.groupby("fold").n_gt75.first() / 24).round(0).astype(int).to_dict()}
d["onnx"] = pd.read_csv(R/"onnx_benchmark_y5.csv").to_dict("records")
(ROOT/"report"/"data.json").write_text(json.dumps(d, ensure_ascii=False, default=float), encoding="utf-8")
print("ok", (ROOT/"report"/"data.json").stat().st_size)
