"""실시간 예측 사후 검증: results/live_log.jsonl 에 남은 예측을, 시간이 지난 뒤 에어코리아 실제값과 비교한다.

각 기록(예측 시점 t 의 24시간 예측)에 대해 MAE 를 두 가지 단순한 기준과 같이 낸다.
  - persistence: t 시점 값을 24시간 내내 그대로 유지
  - 어제 같은 시각: 목표 시각 - 24시간 의 실제값
실제값은 에어코리아에서 서울 25개 구 평균으로 다시 받아 만든다(결측이 6시간 이하면 선형 보간).
기록이 한 건이면 표본이 하나뿐이라, 정확도를 말하는 근거가 아니라 "돌아가는지 확인"으로만 쓴다.

실행: python src/09_live_check.py [--hours 168]
산출: results/live_check.md
"""
import argparse
import json
from datetime import datetime, timedelta

import numpy as np

import airkorea
from data_split import ROOT

LOG = ROOT / "results" / "live_log.jsonl"


def load_log():
    return [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]


def evaluate(rec, actual):
    """actual: {시각: 실제값}. 가용한 horizon 만으로 평가한다."""
    end = datetime.fromisoformat(rec["end_time"])
    pred, last = np.array(rec["pm25_pred"]), rec["history_pm25"][-1]
    rows = []
    for k in range(1, 25):
        t = end + timedelta(hours=k)
        if t in actual and (t - timedelta(hours=24)) in actual:
            rows.append((k, pred[k - 1], actual[t], last, actual[t - timedelta(hours=24)]))
    if not rows:
        return None
    k, p, a, per, nv = (np.array(c) for c in zip(*rows))
    return {"end_time": rec["end_time"], "n_hours": len(k), "model_mae": float(np.abs(p - a).mean()), "persistence_mae": float(np.abs(per - a).mean()),
            "naive_mae": float(np.abs(nv - a).mean()), "bias": float((p - a).mean()), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=168, help="실제값을 몇 시간 전까지 받아 올지")
    args = ap.parse_args()
    recs = load_log()
    r = airkorea.seoul_hourly(hours=args.hours)
    stamps = [r["end_time"] - timedelta(hours=args.hours - 1 - i) for i in range(args.hours)]
    actual = dict(zip(stamps, r["values"]))
    print(f"기록 {len(recs)}건 | 실제값 {stamps[0]:%m/%d %H시} ~ {stamps[-1]:%m/%d %H시} (측정소 {r['stations_ok']}/25)")
    out = ["# 실시간 예측 사후 검증", "", f"실제값 구간: {stamps[0]:%Y-%m-%d %H시} ~ {stamps[-1]:%Y-%m-%d %H시}", ""]
    for rec in recs:
        ev = evaluate(rec, actual)
        if ev is None:
            print(rec["end_time"], "→ 실제값 구간 밖이거나 아직 시간이 지나지 않음")
            continue
        line = (f"{ev['end_time']} 기준 {ev['n_hours']}시간 | 모델 MAE {ev['model_mae']:.2f} | persistence {ev['persistence_mae']:.2f} | "
                f"어제 같은 시각 {ev['naive_mae']:.2f} | 편향(예측-실제) {ev['bias']:+.2f}")
        print(line)
        out += [f"## {ev['end_time']} 기준", "", line, "", "| 시간 뒤 | 예측 | 실제 | persistence | 어제 같은 시각 |", "|---|---|---|---|---|"]
        out += [f"| {int(k)} | {p:.1f} | {a:.1f} | {per:.1f} | {nv:.1f} |" for k, p, a, per, nv in ev["rows"]]
        out.append("")
    (ROOT / "results" / "live_check.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
