"""기상청 동네예보(단기예보) 발표본 수집. 대결 상대의 예보 원본을 쌓아 둔다.

과거 발표본은 API가 며칠치만 주므로 정기적으로 돌려 누적해야 한다.
(.github/workflows/collect_forecast.yml 이 3시간마다 실행)
env: DATA_GO_KR_KEY
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import BASE_TIMES, RAW, SITES  # noqa: E402

URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
KEY = os.environ["DATA_GO_KR_KEY"]
KST = timezone(timedelta(hours=9))


def latest_base(now):
    """API 제공 지연(발표 후 약 10분)을 고려해 가장 최근 발표 시각을 구한다."""
    t = now - timedelta(minutes=15)
    for bt in reversed(BASE_TIMES):
        if t.strftime("%H%M") >= bt:
            return t.strftime("%Y%m%d"), bt
    y = t - timedelta(days=1)
    return y.strftime("%Y%m%d"), BASE_TIMES[-1]


def fetch(nx, ny, base_date, base_time):
    r = requests.get(URL, timeout=60, params={
        "serviceKey": KEY, "pageNo": 1, "numOfRows": 1000, "dataType": "JSON",
        "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny,
    })
    r.raise_for_status()
    return r.json()["response"]["body"]["items"]["item"]


def main():
    base_date, base_time = latest_base(datetime.now(KST))
    out_dir = RAW / "kma_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in SITES.items():
        df = pd.DataFrame(fetch(s["nx"], s["ny"], base_date, base_time))
        path = out_dir / f"{name}_{base_date}_{base_time}.csv"
        df.to_csv(path, index=False)
        print(path.name, len(df), "rows")


if __name__ == "__main__":
    main()
