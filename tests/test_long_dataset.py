"""장기 데이터(2018-01 ~ 2026-09)가 이어 붙여진 결과로서 올바른지, 그리고 실험 설정이 이전 결과를 재현하는지 확인한다."""
import numpy as np
import pandas as pd
import pytest
from conftest import load_script

from data_split import ROOT, load

LONG = ROOT / "data" / "processed" / "seoul_hourly_long.csv"
pytestmark = pytest.mark.skipif(not LONG.exists(), reason="장기 데이터가 없습니다 (src/14_build_long_dataset.py)")


@pytest.fixture(scope="module")
def long_df():
    return load(LONG)


def test_long_dataset_is_continuous_hourly(long_df):
    assert long_df.index.min() == pd.Timestamp("2018-01-01 00:00") and long_df.index.max() == pd.Timestamp("2026-09-30 23:00")
    assert len(long_df) == 76680
    assert (long_df.index.to_series().diff().dropna() == np.timedelta64(1, "h")).all()
    assert long_df.isna().sum().sum() == 0


def test_long_dataset_agrees_with_original_on_overlap(long_df):
    """이어 붙인 데이터의 2024~2025 구간은 원래 전처리 결과와 값이 같아야 한다(기존 결과를 건드리지 않았다는 확인)."""
    orig = load()
    assert np.allclose(long_df.loc[orig.index, "PM25"].values, orig["PM25"].values)
    assert np.allclose(long_df.loc[orig.index, "temp"].values, orig["temp"].values)


def test_setting_A_reproduces_original_training_set(long_df):
    """15_longer_data 의 설정 A(2024 학습)는 원래 설정과 같은 학습 샘플 수(8,424)를 만든다."""
    m = load_script("ld", "15_longer_data.py")
    df = long_df.loc["2024-01-01":]
    assert len(m.origins(df, *m.SPLITS["A"]["train"])) == 8424
    # 검증/테스트 구간은 서로 겹치지 않고 시간순
    S = {k: m.origins(df, *v) for k, v in m.SPLITS["A"].items()}
    assert S["train"].max() < S["val"].min() < S["T1"].min() < S["T2"].min()


def test_new_test_period_is_after_everything_used_for_training(long_df):
    m = load_script("ld", "15_longer_data.py")
    for setting in ("A", "B"):
        s = m.SPLITS[setting]
        assert pd.Timestamp(s["train"][1]) < pd.Timestamp(s["val"][0]) < pd.Timestamp(s["T1"][0]) < pd.Timestamp(s["T2"][0])
