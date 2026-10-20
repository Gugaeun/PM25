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
WORKERS = (1, 2, 4)
LEVELS = [(1, 300), (4, 400), (16, 800), (32, 1200), (64, 1600)]  # (동시 사용자 수, 총 요청 수)
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
    rows = []
    with mp.Pool(PROCS) as ex:
        for w in WORKERS:
            srv = start_server(w)
            try:
                run(4, 80, pool, ex)  # 예열(결과에 넣지 않음)
                for lv, n in LEVELS:
                    r = run(lv, n, pool, ex)
                    rows.append({"worker": w, **r})
                    print(rows[-1], flush=True)
            finally:
                stop_server(srv)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "load_test.csv", index=False, encoding="utf-8-sig")
    md = ("# API 부하 테스트 (POST /predict)\n\n" + df.to_markdown(index=False) +
          f"\n\n같은 컴퓨터(논리 코어 {os.cpu_count()}개)에서 클라이언트(프로세스 {PROCS}개)와 서버(uvicorn, ONNX Runtime)를 함께 돌린 결과입니다. 운영 환경(별도 서버, 실제 네트워크)의 성능이 아닙니다. "
          "요청 본문은 테스트 구간 200개 시점을 돌려 썼습니다. 같은 조건에서 다시 재면 수치가 꽤 달라질 수 있습니다.\n")
    (ROOT / "results" / "load_test.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
