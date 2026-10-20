"""PyTorch LSTM / 1D-CNN 학습 + 기상 변수 ablation (LightGBM 포함).

- 입력: 최근 L시간(기본 48)의 변수들, 출력: 1~24h 뒤 PM2.5 (24개 동시 예측)
- 출력은 "직전 PM2.5 + 네트워크 보정량" 형태 (잔차 연결), 손실은 L1(MAE)
- 정규화는 학습 구간 통계만 사용. 조기 종료는 검증 구간 MAE 기준.
- 시드 3개 반복 → 지표는 평균±표준편차, 예측은 시드 평균(앙상블)도 저장
- 이어 실행 가능: results/pytorch/<run>.npz 가 있으면 건너뜀

실행: python src/04_pytorch.py [--only models|ablation|window] [--seeds 3]
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from data_split import HORIZONS, ROOT, horizon_table, load, metrics, split_origins

torch.set_num_threads(8)
# 장치: 기본은 CPU(지금까지의 모든 결과와 같은 값을 재현). GPU 는 환경변수 PM25_DEVICE=cuda 로 켠다.
# GPU 와 CPU 는 부동소수점 연산 순서가 달라 같은 시드라도 결과가 아주 조금 다르다 → 한 비교 안에서는 같은 장치로 통일한다.
DEVICE = os.environ.get("PM25_DEVICE", "cpu")
OUT = ROOT / "results" / "pytorch"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "temp_humid": ["temp", "humid"],
    "ws": ["ws"],
    "wind_uv": ["wind_u", "wind_v"],
    "weather": ["temp", "humid", "ws", "wind_u", "wind_v"],
    "pollutants": ["PM10", "CO", "NO2", "O3", "SO2"],
    "all": ["temp", "humid", "ws", "wind_u", "wind_v", "PM10", "CO", "NO2", "O3", "SO2"],
}


def feature_cols(group):
    return ["PM25"] + (GROUPS[group] if group != "base" else [])


def build_inputs(df, cols, L):
    """(N, 시점, 변수) 전체 시계열 텐서 + 시간 인코딩. 정규화 전."""
    X = df[cols].values.astype(np.float32)
    h = df.index.hour.values
    tt = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], 1).astype(np.float32)
    return np.concatenate([X, tt], 1)


def make_set(Xall, y, pos, L):
    idx = pos[:, None] + np.arange(-L + 1, 1)[None, :]
    X = Xall[idx]  # (n, L, F)
    Y = np.stack([y[pos + h] for h in HORIZONS], 1)
    return torch.tensor(X), torch.tensor(Y, dtype=torch.float32)


class LSTMNet(nn.Module):
    def __init__(self, n_in, hidden=64, layers=2, drop=0.1):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 24))

    def forward(self, x):
        o, _ = self.rnn(x)
        return x[:, -1, 0:1] + self.head(o[:, -1])  # x[...,0] = 정규화된 PM2.5


class CNNNet(nn.Module):
    def __init__(self, n_in, ch=64, drop=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2, stride=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1, stride=2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(drop), nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 24))

    def forward(self, x):
        z = self.conv(x.transpose(1, 2)).flatten(1)
        return x[:, -1, 0:1] + self.head(z)


HIGH = 35.0  # "나쁨 이상" 기준(µg/m³). 고농도 가중 손실에서 사용


def make_loss(kind, ymu, ysd):
    """손실 함수. 기본 'l1'은 이전 결과와 같다. 'wl1'은 정답이 HIGH 를 넘는 샘플에 가중치 3, 'q70'은 분위수(0.7) 손실."""
    thr = (HIGH - ymu) / ysd

    def f(pred, target):
        e = target - pred
        if kind == "l1":
            return e.abs().mean()
        if kind == "wl1":
            w = 1.0 + 2.0 * (target > thr).float()
            return (w * e.abs()).mean()
        if kind == "q70":
            return torch.maximum(0.7 * e, -0.3 * e).mean()
        raise ValueError(kind)
    return f


def run_one(model_name, group, L, seed, df, S, max_epochs=60, patience=8, save_to=None, loss="l1"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols = feature_cols(group)
    Xall = build_inputs(df, cols, L)
    tr_end = S["train"].max() + 1
    mu = Xall[:tr_end].mean(0)
    sd = Xall[:tr_end].std(0) + 1e-6
    mu[-2:], sd[-2:] = 0, 1  # 시간 인코딩은 그대로
    Xn = (Xall - mu) / sd
    y = df["PM25"].values.astype(np.float32)
    ymu, ysd = mu[0], sd[0]
    yn = (y - ymu) / ysd
    dev = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # 같은 시드면 같은 GPU 결과
    data = {k: tuple(t.to(dev) for t in make_set(Xn, yn, S[k], L)) for k in S}  # 데이터가 작아서 통째로 장치에 올린다
    n_in = Xn.shape[1]
    net = (LSTMNet(n_in) if model_name == "lstm" else CNNNet(n_in)).to(dev)
    net(data["train"][0][:2])  # LazyLinear 초기화
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    Xtr, Ytr = data["train"]
    lossf = make_loss(loss, ymu, ysd)
    best, best_state, bad, ep_used = 1e9, None, 0, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)  # 순서는 항상 CPU 생성기로 만들어 장치와 무관
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            lv = lossf(net(Xtr[b]), Ytr[b])
            opt.zero_grad()
            lv.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(data["val"][0]), data["val"][1]).item() * ysd  # 'l1' 이면 검증 MAE 와 같다
        sched.step(v)
        ep_used = ep + 1
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    if save_to is not None:  # 서빙/ONNX 용: 가중치 + 정규화 통계 + 입력 사양
        torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "model": model_name, "cols": cols, "L": L,
                    "mu": mu, "sd": sd, "n_in": n_in}, save_to)
    preds = {}
    with torch.no_grad():
        for k in S:  # 기본은 val/test. 평가 구간을 더 넣으면(예: test2) 그것도 같이 예측한다
            if k == "train":
                continue
            p = net(data[k][0]).cpu().numpy() * ysd + ymu
            preds[k] = np.clip(p, 0, None)
    return preds, ep_used


def sklearn_free_lgbm(df, S, group):
    """02_baselines 의 LightGBM 을 같은 기상 변수 그룹으로 실행."""
    spec = importlib.util.spec_from_file_location("bl", ROOT / "src" / "02_baselines.py")
    bl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bl)
    wc = [c for c in feature_cols(group) if c != "PM25"]
    P = bl.lgbm_pred(df, S, wc)
    return {k: P[k] for k in ("val", "test")}


def evaluate(df, S, name, preds_by_seed):
    y = df["PM25"].values
    rows = []
    for k in ("val", "test"):
        Y = np.stack([y[S[k] + h] for h in HORIZONS], 1)
        ms = [metrics(Y, p[k]) for p in preds_by_seed]
        ens = np.mean([p[k] for p in preds_by_seed], 0)
        horizon_table(Y, ens).to_csv(OUT / f"{name}_{k}_horizon.csv")
        np.save(OUT / f"{name}_{k}_ens_pred.npy", ens)
        rows.append({"run": name, "split": k,
                     "MAE": np.mean([m["MAE"] for m in ms]), "MAE_std": np.std([m["MAE"] for m in ms]),
                     "RMSE": np.mean([m["RMSE"] for m in ms]), "RMSE_std": np.std([m["RMSE"] for m in ms]),
                     "n_seeds": len(ms)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "models", "ablation", "window"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    df = load()
    S = split_origins(df)
    L0 = 48

    jobs = []  # (model, group, L)
    if a.only in ("all", "models"):
        jobs += [("lstm", "base", L0), ("cnn", "base", L0), ("cnn", "weather", L0)]
    if a.only in ("all", "ablation"):
        jobs += [("lstm", g, L0) for g in ["base", "temp_humid", "ws", "wind_uv", "weather", "pollutants", "all"]]
        jobs += [("lgbm", g, L0) for g in ["base", "temp_humid", "ws", "wind_uv", "weather", "pollutants", "all"]]
    if a.only in ("all", "window"):
        jobs += [("lstm", "base", L) for L in (24, 96, 168)]
    seen, rows = set(), []
    for m, g, L in jobs:
        name = f"{m}_{g}_L{L}"
        if name in seen:
            continue
        seen.add(name)
        f = OUT / f"{name}.json"
        if f.exists():
            rows += json.loads(f.read_text())
            continue
        t0 = time.time()
        if m == "lgbm":
            runs = [sklearn_free_lgbm(df, S, g)]
            eps = [0]
        else:
            res = [run_one(m, g, L, s, df, S) for s in range(a.seeds)]
            runs, eps = [r[0] for r in res], [r[1] for r in res]
        r = evaluate(df, S, name, runs)
        for x in r:
            x.update(model=m, group=g, window=L, epochs=float(np.mean(eps)), sec=time.time() - t0)
        f.write_text(json.dumps(r))
        rows += r
        print(name, {x["split"]: round(x["MAE"], 3) for x in r}, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "summary_partial.csv", index=False)


if __name__ == "__main__":
    main()
