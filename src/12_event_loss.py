"""고농도 개선 시도: 손실 함수를 바꾸면 "나쁨 이상" 경보 성능이 좋아지는가.

기본 손실은 L1(MAE)이다. L1 은 중앙값을 맞추려 하므로 드문 정점을 낮게 예측하는 경향이 있다. 두 가지를 비교한다.
  - wl1: 정답이 35µg/m³ 를 넘는 샘플의 손실 가중치를 3배로
  - q70: 분위수(0.7) 손실 — 낮게 예측하는 것을 더 크게 벌점
대상은 기상 변수를 넣은 1D-CNN·LSTM(시드 3개 앙상블). 평가는 MAE 와 고농도 탐지 지표(재현율·정밀도·AP)를 같이 본다.
정확도(MAE)를 얼마나 잃고 경보를 얼마나 얻는지의 교환(trade-off)을 보는 것이 목적이다.

실행: python src/12_event_loss.py [--seeds 3]  →  results/event_loss.csv, results/event_loss.md
"""
import argparse
import importlib.util
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from data_split import HORIZONS, ROOT, load, split_origins


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pt = _load("pt", "04_pytorch.py")
ev = _load("ev", "11_event_metrics.py")
OUT = ROOT / "results" / "pytorch"
T = 35.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    df = load()
    S = split_origins(df)
    y = df["PM25"].values
    Y = {s: np.stack([y[S[s] + h] for h in HORIZONS], 1) for s in ("val", "test")}
    preds = {}
    for model in ("cnn", "lstm"):
        preds[(model, "l1")] = {s: np.load(OUT / f"{model}_weather_L48_{s}_ens_pred.npy") for s in ("val", "test")}
        for loss in ("wl1", "q70"):
            files = {s: OUT / f"{model}_weather_L48_{loss}_{s}_ens_pred.npy" for s in ("val", "test")}
            if not all(f.exists() for f in files.values()):
                t0 = time.time()
                runs = [pt.run_one(model, "weather", 48, sd, df, S, loss=loss)[0] for sd in range(args.seeds)]
                for s in ("val", "test"):
                    np.save(files[s], np.mean([r[s] for r in runs], 0))
                print(f"{model} {loss}: {time.time() - t0:.0f}s", flush=True)
            preds[(model, loss)] = {s: np.load(f) for s, f in files.items()}

    rows = []
    for (model, loss), P in preds.items():
        row = {"모델": model, "손실": loss, "MAE(1~24h)": float(np.abs(P["test"] - Y["test"]).mean())}
        a24, p24 = Y["test"][:, 23], P["test"][:, 23]
        for lab, lo, hi in (("나쁨", 35, 75), ("매우나쁨", 75, 1e9)):
            m = (a24 > lo) & (a24 <= hi)
            row[f"24h 편향({lab})"] = float((p24[m] - a24[m]).mean())
        for h in (6, 24):
            sv, av, st, at = P["val"][:, h - 1], Y["val"][:, h - 1], P["test"][:, h - 1], Y["test"][:, h - 1]
            e = at > T
            rec, pre, f1, _ = ev.prf(e, st > T)
            d = ev.best_alarm_threshold(sv, av, T)
            rec2, pre2, f12, _ = ev.prf(e, st > d)
            row.update({f"{h}h 재현율": rec, f"{h}h 정밀도": pre, f"{h}h AP": average_precision_score(e, st),
                        f"{h}h 튜닝재현율": rec2, f"{h}h 튜닝정밀도": pre2})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "event_loss.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(out.round(2).to_string(index=False))
    md = ["# 손실 함수별 고농도 탐지 (기상 변수 포함, 시드 3개 앙상블, 테스트 2025-07~12)", "",
          "| 모델 | 손실 | MAE | 24h 편향(나쁨) | 24h 편향(매우나쁨) | 6h 재현율 | 6h 정밀도 | 6h AP | 24h 재현율 | 24h 정밀도 | 24h AP |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        md.append(f"| {r['모델']} | {r['손실']} | {r['MAE(1~24h)']:.2f} | {r['24h 편향(나쁨)']:+.1f} | {r['24h 편향(매우나쁨)']:+.1f} | {r['6h 재현율']:.2f} | {r['6h 정밀도']:.2f} | {r['6h AP']:.2f} | {r['24h 재현율']:.2f} | {r['24h 정밀도']:.2f} | {r['24h AP']:.2f} |")
    (ROOT / "results" / "event_loss.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
