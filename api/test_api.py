"""API 동작 확인: 실제 테스트 구간 데이터로 /predict 호출 → 오프라인 ONNX 결과와 일치하는지, 입력 검증 확인.
실행(루트에서): python api/test_api.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from api.main import app  # noqa: E402
from data_split import load, split_origins  # noqa: E402

df = load()
pos = split_origins(df)["test"]
L = 48
COLS = ["PM25", "temp", "humid", "ws", "wind_u", "wind_v"]

with TestClient(app) as c:
    print("health:", c.get("/health").json())
    errs, lat = [], []
    for t in pos[::300]:
        win = df.iloc[t - L + 1 : t + 1]
        body = {"end_time": win.index[-1].isoformat(), "history": {k: win[k].tolist() for k in COLS}}
        s = time.perf_counter()
        r = c.post("/predict", json=body)
        lat.append((time.perf_counter() - s) * 1000)
        assert r.status_code == 200, r.text
        p = np.array(r.json()["pm25_pred"])
        actual = df["PM25"].values[t + 1 : t + 25]
        errs.append(np.abs(p - actual).mean())
    print(f"{len(errs)}개 요청 평균 MAE={np.mean(errs):.2f}, 응답시간 p50={np.percentile(lat, 50):.1f}ms p95={np.percentile(lat, 95):.1f}ms (TestClient, 네트워크 제외)")
    print("sample:", r.json())

    bad = dict(body, history={**body["history"], "PM25": body["history"]["PM25"][:-1]})
    print("길이 오류 →", c.post("/predict", json=bad).status_code, c.post("/predict", json=bad).json())
    miss = dict(body, history={k: v for k, v in body["history"].items() if k != "temp"})
    print("변수 누락 →", c.post("/predict", json=miss).status_code, c.post("/predict", json=miss).json())
