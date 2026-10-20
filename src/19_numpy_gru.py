"""NumPy 로 직접 구현한 GRU (forward + BPTT backward) 와 gradient check.

딥러닝 라이브러리 없이 numpy 만 쓴다. 1) 수치 미분(중앙 차분)과 해석적 기울기가 맞는지 확인하고(gradient check),
2) 같은 모델을 실제 데이터(입력 24시간, 1~24시간 뒤 예측)로 짧게 학습해 persistence 와 비교한다.

한계(공정 비교가 안 되는 이유): 윈도우 24시간, 은닉 32, MSE 손실, 시드 1개, 튜닝 없음이라 PyTorch 실험(윈도우 48, L1, 시드 3개 앙상블)과
조건이 달라 숫자를 나란히 비교할 수 없다. 이 스크립트의 목적은 성능이 아니라 구현의 정확성(기울기 검사)이다.
실행: python src/19_numpy_gru.py → results/numpy_gru.md
"""
import time

import numpy as np

from data_split import ROOT, load, split_origins

L, H, OUT = 24, 32, 24
COLS = ["PM25", "temp", "humid", "ws", "wind_u", "wind_v"]
D = len(COLS)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def init(rng, dtype=np.float64):
    s = lambda *shape: (rng.standard_normal(shape) * np.sqrt(1.0 / shape[0])).astype(dtype)
    return {"Wz": s(D, H), "Uz": s(H, H), "bz": np.zeros(H, dtype), "Wr": s(D, H), "Ur": s(H, H), "br": np.zeros(H, dtype),
            "Wn": s(D, H), "Un": s(H, H), "bn": np.zeros(H, dtype), "Wo": s(H, OUT), "bo": np.zeros(OUT, dtype)}


def forward(P, X):
    """X: (B, L, D) → 예측 (B, OUT), 역전파용 캐시"""
    B = X.shape[0]
    h = np.zeros((B, H), X.dtype)
    cache = []
    for t in range(X.shape[1]):
        x = X[:, t]
        z = sigmoid(x @ P["Wz"] + h @ P["Uz"] + P["bz"])
        r = sigmoid(x @ P["Wr"] + h @ P["Ur"] + P["br"])
        n = np.tanh(x @ P["Wn"] + (r * h) @ P["Un"] + P["bn"])
        hn = (1 - z) * n + z * h
        cache.append((x, h, z, r, n))
        h = hn
    return h @ P["Wo"] + P["bo"], (cache, h)


def backward(P, cache_h, dY):
    cache, hL = cache_h
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["Wo"] = hL.T @ dY
    G["bo"] = dY.sum(0)
    dh = dY @ P["Wo"].T
    for x, h, z, r, n in reversed(cache):
        dz = dh * (h - n)
        dn = dh * (1 - z)
        dh_prev = dh * z
        dn_pre = dn * (1 - n ** 2)
        G["Wn"] += x.T @ dn_pre
        G["Un"] += (r * h).T @ dn_pre
        G["bn"] += dn_pre.sum(0)
        drh = dn_pre @ P["Un"].T
        dr = drh * h
        dh_prev += drh * r
        dz_pre = dz * z * (1 - z)
        dr_pre = dr * r * (1 - r)
        G["Wz"] += x.T @ dz_pre
        G["Uz"] += h.T @ dz_pre
        G["bz"] += dz_pre.sum(0)
        G["Wr"] += x.T @ dr_pre
        G["Ur"] += h.T @ dr_pre
        G["br"] += dr_pre.sum(0)
        dh_prev += dz_pre @ P["Uz"].T + dr_pre @ P["Ur"].T
        dh = dh_prev
    return G


def loss_fn(P, X, Y):
    pred, ch = forward(P, X)
    return 0.5 * np.mean((pred - Y) ** 2) * 1.0, pred, ch


def gradient_check(P, X, Y, eps=1e-6):
    """중앙 차분 (f(w+eps) - f(w-eps)) / 2eps 와 해석적 기울기의 상대 오차(파라미터별)와 원소별 최대 절대 오차."""
    loss, pred, ch = loss_fn(P, X, Y)
    G = backward(P, ch, (pred - Y) / pred.size)  # d(0.5*mean((p-y)^2))/dp = (p-y)/N
    rows = []
    for k, w in P.items():
        num = np.zeros_like(w)
        for i in np.ndindex(w.shape):
            old = w[i]
            w[i] = old + eps
            lp = loss_fn(P, X, Y)[0]
            w[i] = old - eps
            lm = loss_fn(P, X, Y)[0]
            w[i] = old
            num[i] = (lp - lm) / (2 * eps)
        # 파라미터 전체 벡터의 상대 오차 ||g - n|| / (||g|| + ||n||). 기울기가 0 에 가까운 원소 하나하나의 상대 오차는 수치 미분의 반올림 오차에 좌우돼 쓰지 않는다.
        rel = np.linalg.norm(G[k] - num) / (np.linalg.norm(G[k]) + np.linalg.norm(num))
        rows.append((k, w.size, float(rel), float(np.abs(G[k] - num).max())))
    return rows


def make_xy(df, origins, mu, sd):
    vals = ((df[COLS].values - mu) / sd).astype(np.float64)
    pm = df["PM25"].values
    X = np.stack([vals[t - L + 1: t + 1] for t in origins])
    Y = np.stack([pm[t + 1: t + 1 + OUT] for t in origins])
    last = pm[origins]
    return X, Y, last


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    df = load()
    S = split_origins(df)
    tr_rows = df.index <= df.index[S["train"][-1] + OUT]
    mu, sd = df[COLS].values[tr_rows].mean(0), df[COLS].values[tr_rows].std(0)
    ysd = float(sd[0])
    Xtr, Ytr, Ltr = make_xy(df, S["train"], mu, sd)
    Xva, Yva, Lva = make_xy(df, S["val"], mu, sd)
    Xte, Yte, Lte = make_xy(df, S["test"], mu, sd)
    res = lambda Y, last: (Y - last[:, None]) / ysd  # 마지막 PM2.5 대비 변화량(정규화)

    # 1) gradient check: 작은 배치, float64
    P = init(rng)
    idx = rng.choice(len(Xtr), 4, replace=False)
    Xg, Yg = Xtr[idx][:, -8:], res(Ytr[idx], Ltr[idx])  # 시퀀스 길이 8
    rows = gradient_check(P, Xg, Yg)
    worst = max(r[2] for r in rows)
    lines = ["| 파라미터 | 개수 | 상대 오차 | 원소별 최대 절대 오차 |", "|---|---:|---:|---:|"] + [f"| {k} | {n} | {e:.2e} | {g:.2e} |" for k, n, e, g in rows]
    print("\n".join(lines))
    print(f"gradient check 상대 오차(최대) {worst:.2e}", flush=True)
    assert worst < 1e-5, "기울기가 수치 미분과 맞지 않습니다"

    # 2) 짧은 학습 (Adam, MSE, 마지막 PM2.5 대비 변화량을 예측)
    P = init(rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(w) for k, w in P.items()}
    step, best, bestP, bad = 0, 1e9, None, 0
    pers = lambda Y, last: np.abs(Y - last[:, None]).mean()
    for ep in range(40):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), 128):
            b = perm[i: i + 128]
            pred, ch = forward(P, Xtr[b])
            Yb = res(Ytr[b], Ltr[b])
            G = backward(P, ch, (pred - Yb) / pred.size)
            step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * G[k]
                v[k] = 0.999 * v[k] + 0.001 * G[k] ** 2
                P[k] -= 2e-3 * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        pv = forward(P, Xva)[0] * ysd + Lva[:, None]
        mae = np.abs(pv - Yva).mean()
        if mae < best - 1e-4:
            best, bestP, bad = mae, {k: w.copy() for k, w in P.items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"epoch {ep:2d} 검증 MAE {mae:.3f}", flush=True)
        if bad >= 6:
            break
    pt = forward(bestP, Xte)[0] * ysd + Lte[:, None]
    mae_t, mae_p = np.abs(pt - Yte).mean(), pers(Yte, Lte)
    msg = (f"검증 MAE {best:.2f} (persistence {pers(Yva, Lva):.2f}), 테스트 MAE {mae_t:.2f} (persistence {mae_p:.2f}), "
           f"persistence 대비 {(1 - mae_t / mae_p) * 100:.1f}%")
    print(msg)
    md = ("# NumPy GRU 직접 구현\n\n## gradient check (float64, 4개 샘플, 시퀀스 길이 8, 은닉 32, 중앙 차분 eps=1e-6)\n\n" + "\n".join(lines)
          + f"\n\n상대 오차 최대 {worst:.2e} (기준 1e-5 미만이면 통과).\n\n## 짧은 학습 (윈도우 24시간, MSE, Adam, 시드 1개, 조기 종료)\n\n{msg}\n\n"
          "PyTorch 실험(윈도우 48, L1 손실, 시드 3개 앙상블)과 조건이 달라 숫자를 직접 비교할 수 없습니다. 목적은 기울기 계산의 정확성 확인입니다.\n")
    (ROOT / "results" / "numpy_gru.md").write_text(md, encoding="utf-8")
    print(f"총 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
