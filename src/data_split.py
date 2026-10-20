"""시간순 분할 + 지표 공통 모듈 (섹션 6-3 확정 분할).

학습: 2024 전체 / 검증: 2025-01~06 / 테스트: 2025-07~12
샘플의 소속은 "예측 시작 시점(origin) t"로 정한다. t 시점까지의 값만 입력으로 쓰고
t+1..t+24 를 예측한다. 학습 샘플은 타깃(t+24)이 학습 구간을 넘지 않도록 잘라
검증/테스트 값이 학습에 새어 들어가지 않게 한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "seoul_hourly.csv"
HORIZONS = np.arange(1, 25)
SEED = 42
MIN_HISTORY = 336

BOUNDS = {
    "train": ("2024-01-01 00:00", "2024-12-31 23:00"),
    "val": ("2025-01-01 00:00", "2025-06-30 23:00"),
    "test": ("2025-07-01 00:00", "2025-12-31 23:00"),
}


def load(path=None) -> pd.DataFrame:
    """전처리된 시간 데이터. path 를 주면 다른 파일(예: 장기 데이터)을 읽는다."""
    return pd.read_csv(path or DATA, encoding="utf-8-sig", parse_dates=["datetime"]).set_index("datetime")


def split_origins(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """split별 origin의 정수 위치. 타깃 t+24 가 데이터 끝/구간 끝을 넘으면 제외."""
    idx = df.index
    out = {}
    for name, (a, b) in BOUNDS.items():
        pos = np.where((idx >= a) & (idx <= b))[0]
        # 학습은 타깃이 구간 안에 있어야 함(누수 방지). 검증/테스트도 동일 규칙 적용.
        pos = pos[pos + 24 <= np.where(idx <= b)[0].max()]
        out[name] = pos[pos >= MIN_HISTORY]  # 168h lag / 336h ARIMA 창이 필요해 학습 앞부분 2주는 입력용으로만 사용
    return out


def metrics(y_true, y_pred) -> dict:
    e = np.asarray(y_pred) - np.asarray(y_true)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2)))}


def horizon_table(Y: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    """Y, P: (n_origin, 24). horizon별 MAE/RMSE."""
    return pd.DataFrame(
        [{"horizon": h, **metrics(Y[:, h - 1], P[:, h - 1])} for h in HORIZONS]
    ).set_index("horizon")
