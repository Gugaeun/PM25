"""에어코리아(한국환경공단) 실시간 대기오염정보 OpenAPI 클라이언트.

인증키는 환경변수 AIRKOREA_KEY 에서 읽는다(코드/레포에 쓰지 않는다).
서비스: 한국환경공단_에어코리아_대기오염정보 (B552584/ArpltnInforInqireSvc)
오퍼레이션: getMsrstnAcctoRltmMesureDnsty (측정소별 실시간 측정정보)

seoul_hourly(): 서울 25개 자치구 측정소의 최근 PM2.5 를 시간별로 모아 평균을 낸다.
학습 데이터(서울시 25개 구 평균)와 같은 방식이다. 값은 에어코리아의 확정 전 실시간 자료다.
"""
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

BASE = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"

STATIONS = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구",
            "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]


CALL_LOG = pathlib.Path(__file__).resolve().parents[1] / "results" / "airkorea_calls.json"  # 날짜별 호출 횟수(개발 키는 하루 500건 한도)
_call_lock = threading.Lock()


def calls_today() -> int:
    try:
        return int(json.loads(CALL_LOG.read_text(encoding="utf-8")).get(datetime.now().strftime("%Y-%m-%d"), 0))
    except (OSError, ValueError):
        return 0


def _count_call() -> None:
    """에어코리아에 요청을 보낼 때마다 오늘 호출 횟수를 1 올린다(수집기와 /live 화면이 같은 한도를 쓰므로 파일에 모은다)."""
    with _call_lock:
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            d = json.loads(CALL_LOG.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        d = {today: int(d.get(today, 0)) + 1}  # 지난 날짜 기록은 버린다
        try:
            CALL_LOG.write_text(json.dumps(d), encoding="utf-8")
        except OSError:
            pass


def get_key() -> str:
    key = os.environ.get("AIRKOREA_KEY")
    if not key and sys.platform == "win32":  # setx 로 저장했지만 이 프로세스가 그 전에 시작된 경우
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = winreg.QueryValueEx(k, "AIRKOREA_KEY")[0]
        except OSError:
            key = None
    if not key:
        raise RuntimeError("환경변수 AIRKOREA_KEY 가 없습니다.")
    return urllib.parse.unquote(key) if "%" in key else key  # Encoding 키가 들어와도 처리


def fetch_station(name: str, rows: int = 5, term: str = "DAILY", timeout: int = 12, attempts: int = 3) -> dict:
    q = urllib.parse.urlencode({"serviceKey": get_key(), "returnType": "json", "numOfRows": rows, "pageNo": 1,
                                "stationName": name, "dataTerm": term, "ver": "1.3"})
    last = None
    for i in range(attempts):  # 공공 API 는 가끔 일시적으로 느리거나 5xx 를 준다 → 짧게 재시도
        _count_call()
        try:
            with urllib.request.urlopen(f"{BASE}?{q}", timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code < 500:  # 키 오류 등 4xx 는 재시도해도 소용없다
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(0.5 * (i + 1))
    raise last


def parse_time(s: str) -> datetime:
    """'2026-10-16 15:00' → datetime. 에어코리아는 자정을 '24:00'(전날 날짜)으로 줄 때가 있다."""
    d, t = s.split(" ")
    h = int(t[:2])
    base = datetime.strptime(d, "%Y-%m-%d")
    return base + timedelta(hours=h)  # 24:00 은 다음날 00:00 이 된다


def station_values(name: str, rows: int, fetch=fetch_station) -> dict:
    """측정소 하나의 {시각: PM2.5}. 결측('-')이거나 측정값 플래그가 붙은 값은 뺀다."""
    d = fetch(name, rows=rows, term="MONTH")
    items = d["response"]["body"]["items"] or []
    out = {}
    for it in items:
        v, flag = it.get("pm25Value"), it.get("pm25Flag")
        if flag or v in (None, "", "-"):
            continue
        try:
            out[parse_time(it["dataTime"])] = float(v)
        except (ValueError, KeyError):
            continue
    return out


def seoul_hourly(hours: int = 48, min_stations: int = 12, max_gap: int = 6, fetch=fetch_station) -> dict:
    """서울 25개 구 평균 PM2.5 의 최근 `hours`시간.

    - 한 시각의 평균은 값이 있는 측정소가 min_stations 개 이상일 때만 인정한다.
    - 가장 최근 시각은 아직 일부 측정소만 올라온 경우가 있어, 조건을 만족하는 마지막 시각을 끝으로 쓴다.
    - 중간 결측은 max_gap 시간까지만 선형 보간한다(전처리와 같은 규칙).
    """
    def one(name):
        try:
            return station_values(name, rows=hours + 24, fetch=fetch), None
        except Exception as e:  # noqa: BLE001
            return {}, e

    with ThreadPoolExecutor(3) as ex:  # 동시에 많이 부르면 일부 요청이 거절돼서 3개로 제한
        results = list(ex.map(one, STATIONS))
    errors = [e for _, e in results if e is not None]
    if len(errors) == len(STATIONS):
        raise errors[0]
    sums, cnts = {}, {}
    for vals, _ in results:
        for t, v in vals.items():
            sums[t] = sums.get(t, 0.0) + v
            cnts[t] = cnts.get(t, 0) + 1
    ok = [t for t, c in cnts.items() if c >= min_stations]
    if not ok:
        raise ValueError("평균을 낼 수 있을 만큼 측정소 값이 모이지 않았습니다.")
    end = max(ok)
    stamps = [end - timedelta(hours=hours - 1 - i) for i in range(hours)]
    series = [sums[t] / cnts[t] if cnts.get(t, 0) >= min_stations else None for t in stamps]
    if series[0] is None:
        raise ValueError("48시간 구간의 시작 값이 없어 보간할 수 없습니다.")
    i = 0
    while i < hours:  # 선형 보간
        if series[i] is None:
            j = i
            while j < hours and series[j] is None:
                j += 1
            if j == hours or j - i > max_gap:
                raise ValueError(f"결측이 {max_gap}시간을 넘거나 끝에 있어 보간할 수 없습니다.")
            a, b = series[i - 1], series[j]
            for k in range(i, j):
                series[k] = a + (b - a) * (k - i + 1) / (j - i + 1)
            i = j
        i += 1
    return {"end_time": end, "values": [round(v, 2) for v in series], "n_stations": [cnts.get(t, 0) for t in stamps],
            "stations_ok": len(STATIONS) - len(errors), "stations_failed": len(errors)}


if __name__ == "__main__":
    # 시험 호출: 종로구 최근 5건, 그리고 서울 평균. 인증키는 출력하지 않는다.
    try:
        d = fetch_station("종로구")
        print("header:", d["response"]["header"])
        for it in d["response"]["body"]["items"][:5]:
            print({k: it.get(k) for k in ("dataTime", "pm25Value", "pm25Flag", "pm10Value", "o3Value")})
        r = seoul_hourly()
        print("서울 평균 끝 시각:", r["end_time"], "| 측정소 성공/실패:", r["stations_ok"], r["stations_failed"])
        print("최근 8시간 평균:", r["values"][-8:], "| 시각별 측정소 수:", r["n_stations"][-8:])
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.reason, "|", e.read().decode("utf-8", "replace")[:300])
        sys.exit(1)
