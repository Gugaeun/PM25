"""PM2.5 예측 API (FastAPI + ONNX Runtime). torch 없이 동작.

실행: uvicorn api.main:app --port 8000   (프로젝트 루트에서)
모델 교체: 환경변수 MODEL_NAME (기본 cnn_weather_L48_y5 = 학습 5년, 또는 lstm_weather_L48_y5, 1년 학습 모델은 _y5 없는 이름)

POST /predict
  {"end_time": "2025-12-31T23:00:00",           # history 마지막 값의 시각 (시간 단위)
   "history": {"PM25": [...48개], "temp": [...], "humid": [...], "ws": [...], "wind_u": [...], "wind_v": [...]}}
  → 1~24시간 뒤 PM2.5 예측. 값은 시간순(오래된 것 → 최신).
"""
import csv
import json
import os
import sys
import threading
import time
import urllib.error
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import airkorea  # noqa: E402  (실시간 수집, 인증키는 환경변수 AIRKOREA_KEY)

MODELS = ROOT / "models"
LIVE_MODEL = os.environ.get("LIVE_MODEL_NAME", "cnn_base_L48_y5")  # 기상 변수 없이 PM2.5만 쓰는 모델
LIVE_TTL = 600  # 실시간 조회 결과를 10분 재사용 (API 호출 수와 대기 시간을 줄이려고)
DEMO_FROM = "2025-07-01 00:00:00"  # 데모는 테스트 구간(학습·검증에 쓰지 않은 기간)만 허용
MODEL_NAME = os.environ.get("MODEL_NAME", "cnn_weather_L48_y5")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    meta = json.loads((MODELS / f"{MODEL_NAME}.json").read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    state["meta"] = meta
    state["sess"] = ort.InferenceSession(str(MODELS / f"{MODEL_NAME}.onnx"), so, providers=["CPUExecutionProvider"])
    state["mu"] = np.array(meta["mu"], dtype=np.float32)
    state["sd"] = np.array(meta["sd"], dtype=np.float32)
    if (MODELS / f"{LIVE_MODEL}.onnx").exists():
        lm = json.loads((MODELS / f"{LIVE_MODEL}.json").read_text(encoding="utf-8"))
        state["live"] = {"meta": lm, "mu": np.array(lm["mu"], dtype=np.float32), "sd": np.array(lm["sd"], dtype=np.float32),
                         "sess": ort.InferenceSession(str(MODELS / f"{LIVE_MODEL}.onnx"), so, providers=["CPUExecutionProvider"])}
    with open(ROOT / "data" / "processed" / "seoul_hourly.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))  # 데모용: 과거 데이터를 "현재처럼" 꺼내 쓰기 위해 보관
    state["times"] = [r["datetime"] for r in rows]
    state["idx"] = {t: i for i, t in enumerate(state["times"])}
    state["data"] = {c: [float(r[c]) for r in rows] for c in meta["cols"]}
    yield
    state.clear()


app = FastAPI(title="Seoul PM2.5 forecast", version="0.1", lifespan=lifespan)


class PredictRequest(BaseModel):
    end_time: datetime = Field(description="history 마지막 값의 시각(KST, 시간 단위)")
    history: dict[str, list[float]] = Field(description="변수명 → 최근 L시간 값(오래된 것부터)")


class PredictResponse(BaseModel):
    model: str
    horizons_h: list[int]
    pm25_pred: list[float]
    mean_next_24h: float


@app.get("/health")
def health():
    m = state["meta"]
    return {"status": "ok", "model": MODEL_NAME, "input_cols": m["cols"], "window": m["L"],
            "live_model": LIVE_MODEL if "live" in state else None}


def _run(meta, sess, mu, sd, history, end_time):
    """입력 검증 → 정규화 → ONNX 실행 → 원래 단위로 복원한 24시간 예측."""
    L = meta["L"]
    missing = [c for c in meta["cols"] if c not in history]
    if missing:
        raise HTTPException(422, f"missing variables: {missing}")
    for c in meta["cols"]:
        v = history[c]
        if len(v) != L:
            raise HTTPException(422, f"'{c}' needs exactly {L} values, got {len(v)}")
        if not np.all(np.isfinite(v)):
            raise HTTPException(422, f"'{c}' contains NaN/inf")
    X = np.stack([history[c] for c in meta["cols"]], 1).astype(np.float32)
    hours = (end_time.hour + np.arange(-L + 1, 1)) % 24
    tt = np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], 1).astype(np.float32)
    X = (np.concatenate([X, tt], 1) - mu) / sd
    out = sess.run(None, {"x": X[None]})[0][0]
    return np.clip(out * sd[0] + mu[0], 0, None)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pred = _run(state["meta"], state["sess"], state["mu"], state["sd"], req.history, req.end_time)
    return PredictResponse(model=MODEL_NAME, horizons_h=list(range(1, 25)),
                           pm25_pred=[round(float(p), 2) for p in pred], mean_next_24h=round(float(pred.mean()), 2))


# ---------- 실시간 (에어코리아) ----------
_live_lock = threading.Lock()
LIVE_LOG = ROOT / "results" / "live_log.jsonl"


def _log_live(body):
    """새로 받은 실시간 예측을 한 줄씩 남긴다(같은 기준 시각은 한 번만). 나중에 실제값과 비교하려고(src/09_live_check.py)."""
    try:
        LIVE_LOG.parent.mkdir(exist_ok=True)
        if LIVE_LOG.exists():
            last = LIVE_LOG.read_text(encoding="utf-8").strip().splitlines()[-1:]
            if last and json.loads(last[0]).get("end_time") == body["end_time"]:
                return
        keep = {k: body[k] for k in ("model", "end_time", "history_pm25", "pm25_pred", "mean_next_24h", "stations_ok", "fetched_at")}
        with open(LIVE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(keep, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001  기록 실패가 예측 응답을 막으면 안 된다
        pass


@app.get("/live", include_in_schema=False)
def live_page():
    return FileResponse(Path(__file__).parent / "static" / "live.html")


@app.get("/live/predict")
def live_predict(refresh: bool = False):
    """에어코리아에서 서울 25개 구의 최근 48시간 PM2.5 를 받아 지금부터 24시간 뒤까지 예측한다."""
    lm = state.get("live")
    if lm is None:
        raise HTTPException(503, f"live model '{LIVE_MODEL}' not found. run: python src/08_live_model.py")
    with _live_lock:
        c = state.get("live_cache")
        if c and not refresh and time.time() - c["at"] < LIVE_TTL:
            return {**c["body"], "cached": True}
        try:
            r = airkorea.seoul_hourly(hours=lm["meta"]["L"])
        except RuntimeError as e:  # 인증키 환경변수 없음
            raise HTTPException(503, str(e))
        except urllib.error.HTTPError as e:
            raise HTTPException(502, f"AirKorea API error {e.code}")
        except ValueError as e:  # 측정소 값이 부족하거나 결측이 너무 김
            raise HTTPException(502, str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"AirKorea fetch failed: {type(e).__name__}")
        pred = _run(lm["meta"], lm["sess"], lm["mu"], lm["sd"], {"PM25": r["values"]}, r["end_time"])
        body = {"model": LIVE_MODEL, "end_time": r["end_time"].isoformat(), "history_pm25": r["values"],
                "pm25_pred": [round(float(p), 2) for p in pred], "mean_next_24h": round(float(pred.mean()), 2),
                "stations_ok": r["stations_ok"], "stations_failed": r["stations_failed"], "stations_last_hour": r["n_stations"][-1],
                "fetched_at": datetime.now().isoformat(timespec="seconds"), "cached": False}
        state["live_cache"] = {"at": time.time(), "body": body}
        _log_live(body)
        return body


# ---------- 데모 (과거 데이터 재현) ----------
@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(Path(__file__).parent / "static" / "demo.html")


@app.get("/demo/range")
def demo_range():
    """선택 가능한 예측 시점(end_time) 범위. 뒤 24시간의 실제값이 있어야 해서 끝에서 24시간 전까지."""
    t = state["times"]
    return {"first": t[state["idx"][DEMO_FROM]].replace(" ", "T"), "last": t[-25].replace(" ", "T"), "model": MODEL_NAME}


@app.get("/demo/window")
def demo_window(end_time: datetime):
    """end_time 까지의 최근 L시간 입력값과, 이후 24시간의 실제 PM2.5 (비교용)."""
    L = state["meta"]["L"]
    i = state["idx"].get(end_time.strftime("%Y-%m-%d %H:00:00"))
    if i is None or i < state["idx"][DEMO_FROM] or i + 24 >= len(state["times"]):
        raise HTTPException(422, "end_time must be an hour between demo range (see /demo/range)")
    return {"end_time": state["times"][i].replace(" ", "T"),
            "history": {c: v[i - L + 1 : i + 1] for c, v in state["data"].items()},
            "actual_next_24h": state["data"]["PM25"][i + 1 : i + 25]}
