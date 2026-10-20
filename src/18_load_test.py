"""API 반복·동시 요청 부하 테스트: 같은 컴퓨터에서 uvicorn 을 worker 1·2·4개로 띄우고 /predict 를 동시 사용자 수별로 호출한다.

측정: 요청 수, 오류 수, 처리량(요청/초), 응답시간 p50/p95/p99 (클라이언트 기준, 같은 컴퓨터라 네트워크 지연은 거의 없음).
클라이언트는 프로세스 4개(각각 스레드 여러 개)로 나눠 파이썬 GIL 때문에 클라이언트가 먼저 한계에 닿는 일을 줄였다.
한계: 컴퓨터 한 대에서 클라이언트와 서버가 CPU 를 나눠 쓴다(논리 코어 8개). 운영 환경(별도 서버, 실제 네트워크)의 성능이 아니다.
실행: python src/18_load_test.py → results/load_test.csv, results/load_test.md
"""
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_split import load, split_origins  # noqa: E402

PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
COLS = ["PM25", "temp", "humid", "ws", "wind_u", "wind_v"]
WORKERS = tuple(int(x) for x in os.environ.get("LT_WORKERS", "1,2,4").split(","))  # 환경변수로 일부만 다시 잴 수 있다 (LT_WORKERS=4,2 LT_LEVELS=32:1200 LT_REPEATS=6 LT_SUFFIX=_hang)
LEVELS = [tuple(int(v) for v in x.split(":")) for x in os.environ.get("LT_LEVELS", "1:300,4:400,16:800,32:1200").split(",")]  # (동시 사용자 수, 총 요청 수). 64명은 클라이언트가 병목이라 뺀다
REPEATS = int(os.environ.get("LT_REPEATS", "3"))  # 같은 측정을 몇 번 반복할지(한 번 잰 값으로 말하지 않는다)
SUFFIX = os.environ.get("LT_SUFFIX", "")  # 일부만 다시 잰 결과는 파일 이름 뒤에 붙여 본 결과를 덮어쓰지 않는다
PROCS = 4  # 클라이언트 프로세스 수


def bodies(n):
    df = load()
    pos = split_origins(df)["test"]
    out = []
    for t in pos[:: max(1, len(pos) // n)][:n]:
        w = df.iloc[t - 47: t + 1]
        out.append({"end_time": w.index[-1].isoformat(), "history": {k: w[k].tolist() for k in COLS}})
    return out


def client(args):
    """프로세스 하나: threads 개 스레드로 total 건을 보낸다. (지연 목록, 오류 수, 시작·끝 시각)"""
    threads, total, pool, offset = args
    lat, errs, lock = [], [0], threading.Lock()
    idx = iter(range(total))

    def worker():
        with httpx.Client(base_url=URL, timeout=60) as c:
            while True:
                with lock:
                    i = next(idx, None)
                if i is None:
                    return
                b = pool[(offset + i) % len(pool)]
                s = time.perf_counter()
                try:
                    r = c.post("/predict", json=b)
                    ok = r.status_code == 200 and len(r.json()["pm25_pred"]) == 24
                except Exception:
                    ok = False
                ms = (time.perf_counter() - s) * 1000
                with lock:
                    lat.append(ms)
                    errs[0] += 0 if ok else 1

    t0 = time.time()
    ths = [threading.Thread(target=worker) for _ in range(threads)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    return lat, errs[0], t0, time.time()


def run(level, total, pool, ex):
    procs = min(PROCS, level)
    th = [level // procs + (1 if i < level % procs else 0) for i in range(procs)]
    tt = [total // procs + (1 if i < total % procs else 0) for i in range(procs)]
    res = ex.map(client, [(th[i], tt[i], pool, i * 37) for i in range(procs)])
    res = list(res)
    lat = [x for r in res for x in r[0]]
    errs = sum(r[1] for r in res)
    dt = max(r[3] for r in res) - min(r[2] for r in res)
    return {"동시 사용자": level, "요청 수": total, "오류": errs, "처리량(요청/초)": round(total / dt, 1),
            "p50(ms)": round(float(np.percentile(lat, 50)), 1), "p95(ms)": round(float(np.percentile(lat, 95)), 1),
            "p99(ms)": round(float(np.percentile(lat, 99)), 1), "최대(ms)": round(float(max(lat)), 1)}


def start_server(workers):
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--workers", str(workers), "--log-level", "warning"], cwd=ROOT)
    for _ in range(90):
        try:
            if httpx.get(URL + "/health", timeout=2).status_code == 200:
                time.sleep(1.0 + workers * 0.5)  # 모든 worker 가 모델을 불러올 시간
                return srv
        except Exception:
            time.sleep(1)
    srv.terminate()
    raise SystemExit(f"서버(worker {workers})가 뜨지 않았습니다")


def stop_server(srv):
    """uvicorn --workers 는 master 만 끄면 worker 가 남는다 → 프로세스 트리째 끈다."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True)
    srv.wait(timeout=15)


def main():
    pool = bodies(200)
    runs = []
    with mp.Pool(PROCS) as ex:
        for rep in range(1, REPEATS + 1):  # 반복을 바깥에 두어 시간에 따른 컴퓨터 상태 변화가 설정마다 고르게 섞이게 한다
            for w in WORKERS:
                srv = start_server(w)
                try:
                    run(4, 80, pool, ex)  # 예열(결과에 넣지 않음)
                    for lv, n in LEVELS:
                        r = run(lv, n, pool, ex)
                        runs.append({"반복": rep, "worker": w, **r})
                        print(runs[-1], flush=True)
                finally:
                    stop_server(srv)
    raw = pd.DataFrame(runs)
    raw.to_csv(ROOT / "results" / f"load_test_runs{SUFFIX}.csv", index=False, encoding="utf-8-sig")
    g = raw.groupby(["worker", "동시 사용자"])
    agg = g.agg(**{"오류(합)": ("오류", "sum"), "처리량 중앙값": ("처리량(요청/초)", "median"), "처리량 최소": ("처리량(요청/초)", "min"), "처리량 최대": ("처리량(요청/초)", "max"),
                   "p50 중앙값(ms)": ("p50(ms)", "median"), "p95 중앙값(ms)": ("p95(ms)", "median"), "p95 최소(ms)": ("p95(ms)", "min"), "p95 최대(ms)": ("p95(ms)", "max")}).reset_index().round(1)
    agg.to_csv(ROOT / "results" / f"load_test{SUFFIX}.csv", index=False, encoding="utf-8-sig")
    md = (f"# API 부하 테스트 (POST /predict), {REPEATS}회 반복의 중앙값과 범위\n\n" + agg.to_markdown(index=False) +
          f"\n\n같은 컴퓨터(논리 코어 {os.cpu_count()}개)에서 클라이언트(프로세스 {PROCS}개)와 서버(uvicorn, ONNX Runtime)를 함께 돌린 결과입니다. 운영 환경(별도 서버, 실제 네트워크)의 성능이 아닙니다. "
          "같은 측정을 반복한 값의 중앙값이고 최소·최대로 반복 간 편차를 함께 적었습니다. 요청 본문은 테스트 구간 200개 시점을 돌려 썼습니다. 반복별 원자료는 load_test_runs.csv에 있습니다.\n")
    (ROOT / "results" / f"load_test{SUFFIX}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
