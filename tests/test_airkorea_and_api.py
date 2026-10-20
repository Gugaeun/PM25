"""에어코리아 수집 로직과 API 를 네트워크 없이 검증한다(가짜 응답 사용)."""
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import airkorea
import api.main as api_main


def fake_station(values_by_hour, flag=None):
    """측정소 하나의 에어코리아 응답 흉내. values_by_hour: {datetime: str|None}"""
    def fetch(name, rows=5, term="DAILY", **kw):
        items = [{"dataTime": t.strftime("%Y-%m-%d %H:%M"), "pm25Value": v, "pm25Flag": flag} for t, v in values_by_hour.items()]
        return {"response": {"header": {"resultCode": "00"}, "body": {"items": items}}}
    return fetch


def test_parse_time_handles_24h_convention():
    assert airkorea.parse_time("2026-10-16 24:00") == datetime(2026, 10, 17, 0, 0)
    assert airkorea.parse_time("2026-10-16 15:00") == datetime(2026, 10, 16, 15, 0)


def test_station_values_drops_missing_and_flagged():
    t = datetime(2026, 10, 16, 10)
    f = fake_station({t: "12", t + timedelta(hours=1): "-", t + timedelta(hours=2): None})
    assert airkorea.station_values("종로구", 10, fetch=f) == {t: 12.0}
    assert airkorea.station_values("종로구", 10, fetch=fake_station({t: "12"}, flag="점검")) == {}


def _series(end, hours, f=lambda i: 10.0 + i, hole=()):
    return {end - timedelta(hours=hours - 1 - i): str(f(i)) for i in range(hours) if i not in hole}


def test_seoul_hourly_averages_and_interpolates_short_gap():
    end = datetime(2026, 10, 16, 15)
    r = airkorea.seoul_hourly(hours=48, fetch=fake_station(_series(end, 60, hole={30, 31})))
    assert r["end_time"] == end and len(r["values"]) == 48
    assert r["values"][-1] == pytest.approx(69.0)
    # 구멍 난 시각은 앞뒤 값 사이를 직선으로 이은 값
    full = airkorea.seoul_hourly(hours=48, fetch=fake_station(_series(end, 60)))
    assert max(abs(a - b) for a, b in zip(r["values"], full["values"])) < 1e-6


def test_seoul_hourly_rejects_long_gap_and_too_few_stations():
    end = datetime(2026, 10, 16, 15)
    with pytest.raises(ValueError):
        airkorea.seoul_hourly(hours=48, fetch=fake_station(_series(end, 60, hole=set(range(20, 30)))))
    with pytest.raises(ValueError):
        airkorea.seoul_hourly(hours=48, min_stations=30, fetch=fake_station(_series(end, 60)))


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "LIVE_LOG", tmp_path / "live_log.jsonl")
    with TestClient(api_main.app) as c:
        yield c


def test_health_and_input_validation(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["window"] == 48
    ok = {"end_time": "2025-12-31T23:00:00", "history": {c: [10.0] * 48 for c in h["input_cols"]}}
    r = client.post("/predict", json=ok)
    assert r.status_code == 200 and len(r.json()["pm25_pred"]) == 24 and min(r.json()["pm25_pred"]) >= 0
    short = {**ok, "history": {**ok["history"], "PM25": [10.0] * 47}}
    assert client.post("/predict", json=short).status_code == 422
    missing = {**ok, "history": {k: v for k, v in ok["history"].items() if k != "temp"}}
    assert client.post("/predict", json=missing).status_code == 422


def test_live_predict_uses_collector_and_logs_once(client, monkeypatch):
    end = datetime(2026, 10, 16, 15)
    fake = {"end_time": end, "values": [10.0 + (i % 5) for i in range(48)], "n_stations": [25] * 48, "stations_ok": 25, "stations_failed": 0}
    calls = []
    monkeypatch.setattr(api_main.airkorea, "seoul_hourly", lambda hours: calls.append(hours) or fake)
    b = client.get("/live/predict").json()
    assert calls == [48] and b["end_time"] == end.isoformat() and len(b["pm25_pred"]) == 24 and b["cached"] is False
    assert client.get("/live/predict").json()["cached"] is True and len(calls) == 1  # 10분 재사용
    client.get("/live/predict?refresh=true")  # 강제 새로고침해도 같은 기준 시각은 한 번만 기록
    assert len(api_main.LIVE_LOG.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_live_predict_reports_upstream_errors(client, monkeypatch):
    def boom(hours):
        raise ValueError("측정소 값이 부족합니다")
    monkeypatch.setattr(api_main.airkorea, "seoul_hourly", boom)
    r = client.get("/live/predict?refresh=true")
    assert r.status_code == 502 and "부족" in r.json()["detail"]
