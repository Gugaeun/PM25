"""고농도 탐지 지표: "나쁨 이상(>35µg/m³)"과 "매우 나쁨(>75)"을 얼마나 잡아내는가.

MAE 는 평균 오차라서 드문 고농도 사건을 얼마나 놓치는지 보여주지 못한다. 예측값이 기준을 넘으면 "고농도 예측"으로 보고
재현율(실제 고농도 중 잡은 비율)·정밀도(고농도 예측 중 맞은 비율)·F1 을 낸다.
 - default: 예측값 > 기준 이면 경보
 - val-tuned: 경보 임계값을 검증 구간에서 F1 이 최대가 되게 정해 테스트에 적용 (테스트를 보고 고르지 않는다)
 - AP / AUC: 임계값에 의존하지 않는 순위 지표 (예측값을 점수로 사용)
horizon 1·6·12·24h, 그리고 "다음날 평균"(23시 예측의 24시간 평균 > 기준)에 대해 계산한다.

실행: python src/11_event_metrics.py   →  results/event_metrics.csv, results/event_metrics.md
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from data_split import HORIZONS, ROOT, load, split_origins

R, PT = ROOT / "results", ROOT / "results" / "pytorch"
THRESHOLDS = {"나쁨 이상 (>35)": 35.0, "매우 나쁨 (>75)": 75.0}
MODELS = {
    "persistence": ("baselines_{s}_pred_persistence.npy", R),
    "naive": ("baselines_{s}_pred_naive_yesterday.npy", R),
    "LightGBM": ("baselines_{s}_pred_lightgbm.npy", R),
    "LightGBM+기상": ("lgbm_weather_L48_{s}_ens_pred.npy", PT),
    "LSTM": ("lstm_base_L48_{s}_ens_pred.npy", PT),
    "LSTM+기상": ("lstm_weather_L48_{s}_ens_pred.npy", PT),
    "1D-CNN": ("cnn_base_L48_{s}_ens_pred.npy", PT),
    "1D-CNN+기상": ("cnn_weather_L48_{s}_ens_pred.npy", PT),
}


def prf(actual_event, alarm):
    tp = int((actual_event & alarm).sum())
    rec = tp / max(int(actual_event.sum()), 1)
    pre = tp / max(int(alarm.sum()), 1) if alarm.sum() else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre == pre and (pre + rec) > 0 else 0.0
    return rec, pre, f1, tp


def best_alarm_threshold(score, actual, T):
    """검증 구간에서 F1 이 최대가 되는 경보 임계값 (T-25 ~ T+5 범위 탐색)."""
    ev = actual > T
    if ev.sum() == 0:
        return T
    cands = np.arange(T - 25, T + 5.5, 0.5)
    f = [prf(ev, score > c)[2] for c in cands]
    return float(cands[int(np.argmax(f))])


def main():
    df = load()
    S = split_origins(df)
    y = df["PM25"].values
    rows = []
    for split_name in ("val", "test"):
        pos = S[split_name]
    Y = {s: np.stack([y[S[s] + h] for h in HORIZONS], 1) for s in ("val", "test")}
    m23 = {s: df.index[S[s]].hour == 23 for s in ("val", "test")}
    P = {}
    for name, (pat, base) in MODELS.items():
        P[name] = {s: np.load(base / pat.format(s=s)) for s in ("val", "test")}

    cases = [(f"{h}h 뒤", h - 1) for h in (1, 6, 12, 24)] + [("다음날 평균", None)]
    for tname, T in THRESHOLDS.items():
        for cname, hi in cases:
            def get(model, s):
                if hi is None:
                    return P[model][s][m23[s]].mean(1), Y[s][m23[s]].mean(1)
                return P[model][s][:, hi], Y[s][:, hi]
            for model in MODELS:
                sv, av = get(model, "val")
                st, at = get(model, "test")
                ev = at > T
                rec, pre, f1, tp = prf(ev, st > T)
                d = best_alarm_threshold(sv, av, T)
                rec2, pre2, f12, tp2 = prf(ev, st > d)
                ok = ev.sum() > 0 and (~ev).sum() > 0
                rows.append({"기준": tname, "예측": cname, "모델": model, "테스트 고농도 수": int(ev.sum()), "전체 수": len(at),
                             "기본 재현율": rec, "기본 정밀도": pre, "기본 F1": f1,
                             "경보 임계값(검증에서 선택)": d, "튜닝 재현율": rec2, "튜닝 정밀도": pre2, "튜닝 F1": f12,
                             "AP": average_precision_score(ev, st) if ok else np.nan, "AUC": roc_auc_score(ev, st) if ok else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(R / "event_metrics.csv", index=False, encoding="utf-8-sig")

    # 요약 표(마크다운): 나쁨 이상, 6h / 24h / 다음날 평균
    md = ["# 고농도 탐지 지표 (테스트 2025-07~12)", "", "기본 = 예측값이 기준을 넘으면 경보. 튜닝 = 경보 임계값을 검증 구간에서 F1 최대로 정해 적용. AP·AUC 는 임계값과 무관한 순위 지표.", ""]
    for tname in THRESHOLDS:
        for cname in ("6h 뒤", "24h 뒤", "다음날 평균"):
            sub = out[(out["기준"] == tname) & (out["예측"] == cname)]
            n = int(sub["테스트 고농도 수"].iloc[0]); tot = int(sub["전체 수"].iloc[0])
            md += [f"## {tname}, {cname} (고농도 {n} / 전체 {tot})", "",
                   "| 모델 | 기본 재현율 | 기본 정밀도 | 기본 F1 | 튜닝 재현율 | 튜닝 정밀도 | 튜닝 F1 | AP | AUC |", "|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                md.append(f"| {r['모델']} | {r['기본 재현율']:.2f} | {r['기본 정밀도']:.2f} | {r['기본 F1']:.2f} | {r['튜닝 재현율']:.2f} | {r['튜닝 정밀도']:.2f} | {r['튜닝 F1']:.2f} | {r['AP']:.2f} | {r['AUC']:.2f} |")
            md.append("")
    (R / "event_metrics.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 220)
    show = out[(out["기준"] == "나쁨 이상 (>35)") & (out["예측"].isin(["6h 뒤", "24h 뒤", "다음날 평균"]))]
    print(show[["예측", "모델", "테스트 고농도 수", "기본 재현율", "기본 정밀도", "기본 F1", "튜닝 재현율", "튜닝 정밀도", "튜닝 F1", "AP", "AUC"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
