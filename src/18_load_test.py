"""API 반복·동시 요청 부하 테스트: 같은 컴퓨터에서 uvicorn 을 띄우고 /predict 를 동시 사용자 수별로 호출한다.

측정: 요청 수, 오류 수, 처리량(요청/초), 응답시간 p50/p95/p99 (클라이언트 기준, 같은 컴퓨터라 네트워크 지연은 거의 없음).
한계: 컴퓨터 한 대에서 클라이언트와 서버를 같이 돌려 서로 CPU 를 나눠 쓰고, 서버는 worker 1개이다. 운영 환경의 성능이 아니다.
실행: python src/18_load_test.py → results/load_test.csv, results/load_test.md
"""
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
LEVELS = [(1, 300), (4, 400), (16, 800), (32, 800)]  # (동시 사용자 수, 총 요청 수)


def bodies(n):
    df = load()
    pos = split_origins(df)["test"]
    out = []
    for t in pos[:: max(1, len(pos) // n)][:n]:
        w = df.iloc[t - 47: t + 1]
        out.append({"end_time": w.index[-1].isoformat(), "history": {k: w[k].tolist() for k in COLS}})
    return out


def run(level, total, pool):
    lat, errs, lock = [], [0], threading.Lock()
    idx = iter(range(total))

    def worker():
        with httpx.Client(base_url=URL, timeout=30) as c:
            while True:
                with lock:
                    i = next(idx, None)
                if i is None:
                    return
                b = pool[i % len(pool)]
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

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker) for _ in range(level)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    dt = time.perf_counter() - t0
    return {"동시 사용자": level, "요청 수": total, "오류": errs[0], "처리량(요청/초)": round(total / dt, 1),
            "p50(ms)": round(float(np.percentile(lat, 50)), 1), "p95(ms)": round(float(np.percentile(lat, 95)), 1),
            "p99(ms)": round(float(np.percentile(lat, 99)), 1), "최대(ms)": round(float(max(lat)), 1)}


def main():
    pool = bodies(200)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(PORT), "--log-level", "warning"], cwd=ROOT)
    try:
        for _ in range(60):
            try:
                if httpx.get(URL + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("서버가 뜨지 않았습니다")
        run(1, 30, pool)  # 예열(결과에 넣지 않음)
        rows = [run(lv, n, pool) for lv, n in LEVELS]
    finally:
        srv.terminate()
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "load_test.csv", index=False, encoding="utf-8-sig")
    md = "# API 부하 테스트 (POST /predict)\n\n" + df.to_markdown(index=False) + "\n\n같은 컴퓨터에서 클라이언트와 서버(uvicorn worker 1개, ONNX Runtime)를 함께 돌린 결과입니다. 운영 환경의 성능이 아닙니다. 요청 본문은 테스트 구간 200개 시점을 돌려 썼습니다.\n"
    (ROOT / "results" / "load_test.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
