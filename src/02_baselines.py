"""baseline: naive(어제 같은 시각), persistence, ARIMA, LightGBM.

모두 같은 origin / 같은 horizon(1~24h) / 같은 split 에서 평가한다.
결과: results/baselines_{split}_horizon.csv, results/baselines_summary.csv,
      results/baselines_{split}_pred_{model}.npy  (origin x 24)
실행: python src/02_baselines.py [--arima-stride 3]
"""
import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_split import HORIZONS, ROOT, SEED, horizon_table, load, metrics, split_origins

warnings.filterwarnings("ignore")
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
LAGS = [0, 1, 2, 3, 6, 12, 23, 24, 25, 47, 48, 168]  # origin t 기준 과거 오프셋


def targets(y, pos):
    return np.stack([y[pos + h] for h in HORIZONS], axis=1)


def naive_pred(y, pos):
    """target t+h 의 예측 = y[t+h-24] (어제 같은 시각). h=24 는 y[t]."""
    return np.stack([y[pos + h - 24] for h in HORIZONS], axis=1)


def persistence_pred(y, pos):
    return np.repeat(y[pos][:, None], 24, axis=1)


def arima_pred(y, pos, order=(2, 0, 1), window=336, train_end=None):
    """학습 구간에서 파라미터 1회 추정 후, origin마다 최근 window 시간으로 24h 예측."""
    fit = ARIMA(y[:train_end], order=order).fit()
    P = np.empty((len(pos), 24))
    for i, t in enumerate(pos):
        m = ARIMA(y[t - window + 1 : t + 1], order=order).smooth(fit.params)
        P[i] = m.forecast(24)
    return np.clip(P, 0, None)


def make_features(df, pos, cols):
    y = df["PM25"].values
    X = {f"PM25_lag{l}": y[pos - l] for l in LAGS}
    s = df["PM25"]
    for w in (6, 24, 168):
        X[f"PM25_mean{w}"] = s.rolling(w).mean().values[pos]
    for c in cols:
        X[f"{c}_t"] = df[c].values[pos]
    ts = df.index[pos]
    X["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    X["month"] = ts.month
    X["dow"] = ts.dayofweek
    return pd.DataFrame(X)


def lgbm_pred(df, S, weather_cols=()):
    y = df["PM25"].values
    Xs = {k: make_features(df, S[k], weather_cols) for k in S}
    Ys = {k: targets(y, S[k]) for k in S}
    # 시간 보정: month 는 학습(2024)에서 1~12 를 모두 보므로 사용 가능
    P = {k: np.empty_like(Ys[k]) for k in S}
    for h in HORIZONS:
        mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                min_child_samples=40, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, random_state=SEED, verbose=-1)
        mdl.fit(Xs["train"], Ys["train"][:, h - 1],
                eval_set=[(Xs["val"], Ys["val"][:, h - 1])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        for k in S:
            P[k][:, h - 1] = mdl.predict(Xs[k])
    return P


def daily_mean_eval(df, pos, Y, P):
    """origin 23시 → 다음날 24시간 예측의 평균 = 다음날 일평균(A)."""
    m = df.index[pos].hour == 23
    return metrics(Y[m].mean(1), P[m].mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arima-stride", type=int, default=3, help="ARIMA는 느려서 origin을 N시간 간격으로 평가")
    args = ap.parse_args()

    df = load()
    y = df["PM25"].values
    S = split_origins(df)
    train_end = S["train"].max() + 1
    print({k: (len(v), str(df.index[v[0]]), str(df.index[v[-1]])) for k, v in S.items()})

    preds = {"val": {}, "test": {}}
    for k in preds:
        preds[k]["naive_yesterday"] = naive_pred(y, S[k])
        preds[k]["persistence"] = persistence_pred(y, S[k])
    P = lgbm_pred(df, S)
    for k in preds:
        preds[k]["lightgbm"] = P[k]

    rows = []
    for k in preds:
        Y = targets(y, S[k])
        for name, p in preds[k].items():
            np.save(RES / f"baselines_{k}_pred_{name}.npy", p)
            horizon_table(Y, p).to_csv(RES / f"baselines_{k}_horizon_{name}.csv")
            rows.append({"split": k, "model": name, "stride": 1, **metrics(Y, p),
                         **{f"daily_{a}": b for a, b in daily_mean_eval(df, S[k], Y, p).items()}})

    # ARIMA: 같은 origin 부분집합에서 naive 와도 같이 비교하기 위해 stride 적용
    for k in preds:
        sub = S[k][:: args.arima_stride]
        Y = targets(y, sub)
        p = arima_pred(y, sub, train_end=train_end)
        np.save(RES / f"baselines_{k}_pred_arima.npy", p)
        horizon_table(Y, p).to_csv(RES / f"baselines_{k}_horizon_arima.csv")
        for name, pp in (("arima", p), ("naive_yesterday(same origins)", naive_pred(y, sub))):
            rows.append({"split": k, "model": name, "stride": args.arima_stride, **metrics(Y, pp)})

    out = pd.DataFrame(rows)
    # naive 대비 MAE 감소율(전체 origin 기준 행만)
    base = out[out.model == "naive_yesterday"].set_index("split")["MAE"]
    out["MAE_reduction_vs_naive_%"] = out.apply(
        lambda r: 100 * (1 - r.MAE / base[r.split]) if r.stride == 1 else np.nan, axis=1)
    out.to_csv(RES / "baselines_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
