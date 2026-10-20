"""실시간 예측을 매시간 자동으로 기록한다(서버 없이). 기록은 results/live_log.jsonl 에 쌓이고,
src/09_live_check.py 가 시간이 지난 뒤 실제값과 비교한다.

매시간 정각 +N분(기본 10분)에 에어코리아에서 서울 25개 구 값을 받아 PM2.5 전용 모델로 24시간 예측을 만든다.
같은 기준 시각은 한 번만 기록된다. 컴퓨터가 꺼지거나 절전이면 그동안은 기록되지 않는다.

실행: python src/10_live_collect.py [--minute 10] [--once]
종료: 실행한 창에서 Ctrl+C (백그라운드로 켰다면 작업 관리자에서 python 종료)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import onnxruntime as ort

from data_split import ROOT

sys.path.insert(0, str(ROOT))
import airkorea  # noqa: E402
from api.main import LIVE_LOG, _log_live, _run  # noqa: E402

MODEL = "cnn_base_L48_y5"  # 학습 5년(2020~2024) PM2.5 전용 모델 (src/20_serving_models_5y.py)


def load():
    meta = json.loads((ROOT / "models" / f"{MODEL}.json").read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(ROOT / "models" / f"{MODEL}.onnx"), so, providers=["CPUExecutionProvider"])
    return meta, sess, np.array(meta["mu"], dtype=np.float32), np.array(meta["sd"], dtype=np.float32)


def collect_once(m):
    meta, sess, mu, sd = m
    r = airkorea.seoul_hourly(hours=meta["L"])
    pred = _run(meta, sess, mu, sd, {"PM25": r["values"]}, r["end_time"])
    body = {"model": MODEL, "end_time": r["end_time"].isoformat(), "history_pm25": r["values"],
            "pm25_pred": [round(float(p), 2) for p in pred], "mean_next_24h": round(float(pred.mean()), 2),
            "stations_ok": r["stations_ok"], "fetched_at": datetime.now().isoformat(timespec="seconds")}
    _log_live(body)
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minute", type=int, default=10, help="매시간 몇 분에 받을지")
    ap.add_argument("--every", type=int, default=1, help="몇 시간마다 받을지 (에어코리아 개발 키는 하루 500건 한도, 1회 25건이라 1이면 하루 600건으로 한도를 넘는다)")
    ap.add_argument("--budget", type=int, default=360, help="오늘 에어코리아 호출이 이 횟수를 넘으면 그날은 수집을 건너뛴다(한도 500건에서 /live 화면과 재시도 몫을 남긴다)")
    ap.add_argument("--wait-first", action="store_true", help="시작하자마자 받지 않고 다음 정해진 시각까지 기다린다(재시작할 때 호출을 아끼려고)")
    ap.add_argument("--once", action="store_true", help="한 번만 실행하고 종료")
    args = ap.parse_args()
    m = load()
    first = True
    while True:
        try:
            if first and args.wait_first:
                first = False
                raise StopIteration
            first = False
            if airkorea.calls_today() >= args.budget:
                print(f"[{datetime.now():%m-%d %H:%M}] 오늘 호출 {airkorea.calls_today()}건으로 예산({args.budget}건)에 도달해 건너뜁니다", flush=True)
                raise StopIteration
            b = collect_once(m)
            n = sum(1 for _ in open(LIVE_LOG, encoding="utf-8"))
            print(f"[{datetime.now():%m-%d %H:%M}] 기준 {b['end_time']} | 측정소 {b['stations_ok']}/25 | 24h 평균 예측 {b['mean_next_24h']} | 누적 기록 {n}건", flush=True)
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001  일시적인 API 오류로 멈추지 않는다
            print(f"[{datetime.now():%m-%d %H:%M}] 실패: {type(e).__name__} {str(e)[:80]}", flush=True)
        if args.once:
            return
        now = datetime.now()
        nxt = now.replace(minute=args.minute, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(hours=args.every)
        time.sleep(max(30, (nxt - now).total_seconds()))


if __name__ == "__main__":
    main()
