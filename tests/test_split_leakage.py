"""평가 설계 검증: 미래 정보가 학습·입력에 새어 들어가지 않는지.

이 프로젝트의 핵심 주장("시간순 분할, 누수 방지")을 코드로 확인한다.
실행: python -m pytest tests -q
"""
import numpy as np
import pytest
from conftest import load_script

from data_split import BOUNDS, HORIZONS, MIN_HISTORY, load, metrics, split_origins


@pytest.fixture(scope="module")
def df():
    return load()


@pytest.fixture(scope="module")
def S(df):
    return split_origins(df)


def test_data_shape_and_hourly_index(df):
    assert len(df) == 17544  # 730일 x 24시간
    assert df.index.is_monotonic_increasing
    assert (df.index.to_series().diff().dropna() == np.timedelta64(1, "h")).all()  # 빠진 시간 없음
    assert df["PM25"].isna().sum() == 0


def test_splits_do_not_overlap_and_are_time_ordered(S):
    assert S["train"].max() < S["val"].min() < S["val"].max() < S["test"].min()
    assert len(set(S["train"]) & set(S["val"])) == 0 and len(set(S["val"]) & set(S["test"])) == 0


def test_targets_stay_inside_their_own_period(df, S):
    """어떤 split 의 정답(t+1..t+24)도 그 split 의 기간을 넘지 않는다 → 학습 정답이 검증/테스트 기간으로 새지 않는다."""
    for name, (a, b) in BOUNDS.items():
        last_target = df.index[S[name].max() + 24]
        assert last_target <= np.datetime64(b), name
        assert df.index[S[name].min()] >= np.datetime64(a), name


def test_inputs_have_enough_history(S):
    for v in S.values():
        assert v.min() >= MIN_HISTORY


def test_normalization_statistics_use_train_period_only(df, S):
    """04_pytorch.run_one 은 정규화 통계를 Xall[:S['train'].max()+1] 로 계산한다. 그 끝이 검증 시작 이전인지 확인."""
    tr_end = S["train"].max() + 1
    assert df.index[tr_end] < df.index[S["val"].min()]


def features_are_causal(make, df, origins, cols):
    """origin t 의 특징이 t 이후 값에 의존하지 않는가. 각 origin 마다 그 *이후* 전체를 바꿔 보고 특징이 그대로인지 본다."""
    for t in origins:
        base = make(df, np.array([t]), cols).values
        future = df.copy()
        future.iloc[t + 1:, :] = future.iloc[t + 1:, :] + 123.0
        if not np.allclose(base, make(future, np.array([t]), cols).values):
            return False
    return True


def test_lgbm_features_use_only_the_past(df):
    bl = load_script("bl", "02_baselines.py")
    cols = ["temp", "humid", "ws", "wind_u", "wind_v"]
    assert features_are_causal(bl.make_features, df, [400, 1000, 5000, 12000], cols), "origin 이후 값을 바꿨는데 특징이 달라졌다 → 미래 누수"


def test_the_leakage_check_itself_detects_leaks(df):
    """검사가 실제로 누수를 잡아내는지: '다음 시간 PM2.5'를 특징으로 넣은 가짜 함수는 반드시 실패해야 한다."""
    bl = load_script("bl", "02_baselines.py")

    def leaky(d, pos, c):
        f = bl.make_features(d, pos, c)
        f["leak_next_hour"] = d["PM25"].values[pos + 1]
        return f

    assert not features_are_causal(leaky, df, [400, 1000, 5000], ["temp"])


def test_nn_input_windows_end_at_the_origin(df):
    pt = load_script("pt", "04_pytorch.py")
    cols = pt.feature_cols("weather")
    Xall = pt.build_inputs(df, cols, 48)
    y = df["PM25"].values.astype(np.float32)
    t = np.array([1000, 5000])
    X, Y = pt.make_set(Xall, y, t, 48)
    assert np.allclose(X[0, -1, : len(cols)].numpy(), Xall[1000, : len(cols)])  # 창의 마지막 시점 = origin
    mutated = Xall.copy()
    mutated[1001:] += 99.0  # origin 이후를 바꿔도
    X2, _ = pt.make_set(mutated, y, t[:1], 48)
    assert np.allclose(X.numpy()[0], X2.numpy()[0])  # 입력이 같아야 한다
    assert np.allclose(Y[0].numpy(), [y[1000 + h] for h in HORIZONS])  # 정답은 t+1..t+24


def test_baseline_definitions(df, S):
    bl = load_script("bl", "02_baselines.py")
    y = df["PM25"].values
    pos = S["test"][:50]
    naive, pers = bl.naive_pred(y, pos), bl.persistence_pred(y, pos)
    for h in (1, 12, 24):
        assert np.allclose(naive[:, h - 1], y[pos + h - 24])  # 어제 같은 시각
    assert np.allclose(naive[:, 23], pers[:, 0])  # h=24 의 naive 는 직전 값과 같다
    assert np.allclose(pers, y[pos][:, None])


def test_walk_forward_folds_never_train_on_the_future(df):
    wf = load_script("wf", "07_walk_forward.py")
    for _, a, b in wf.FOLDS:
        F = wf.fold_origins(df, a, b)
        assert F["train"].max() + 24 < F["val"].min() and F["val"].max() + 24 < F["test"].min()
        assert df.index[F["test"].max() + 24] <= np.datetime64(b)


def test_metrics_known_values():
    m = metrics([1, 2, 3], [2, 2, 5])
    assert m["MAE"] == pytest.approx(1.0) and m["RMSE"] == pytest.approx(np.sqrt(5 / 3))
