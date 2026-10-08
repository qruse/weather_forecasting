"""기상청 동네예보(단기예보) 발표본 수집. 대결 상대의 예보 원본을 쌓아 둔다.

API Hub 는 최근 3일치 발표본만 주므로 정기적으로 돌려 누적해야 한다.
이미 받은 발표본은 건너뛰므로 여러 번 실행해도 안전하다.

usage: python scripts/collect_kma_forecast.py            # 최근 3일 전부(백필) + 최신
env:   KMA_APIHUB_KEY  (apihub.kma.go.kr authKey)
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import BASE_TIMES, RAW, SITES  # noqa: E402

URL = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getVilageFcst"
KEY = os.environ["KMA_APIHUB_KEY"]
KST = timezone(timedelta(hours=9))


def issued_bases(now, days=3):
    """제공 지연(발표 후 약 10분)을 고려해 이미 발표된 (날짜, 시각) 목록을 오래된 순으로."""
    cutoff = now - timedelta(minutes=15)
    out = []
    for d in range(days, -1, -1):
        day = (now - timedelta(days=d)).strftime("%Y%m%d")
        for bt in BASE_TIMES:
            issued = datetime.strptime(day + bt, "%Y%m%d%H%M").replace(tzinfo=KST)
            if issued <= cutoff and now - issued <= timedelta(days=days, hours=1):
                out.append((day, bt))
    return out


def fetch(nx, ny, base_date, base_time):
    rows, page = [], 1
    while True:
        r = requests.get(URL, timeout=60, params={
            "authKey": KEY, "pageNo": page, "numOfRows": 1000, "dataType": "JSON",
            "base_date": base_date, "base_time": base_time, "nx": nx, "ny": ny,
        })
        r.raise_for_status()
        body = r.json()["response"]
        if body["header"]["resultCode"] != "00":
            print("skip", base_date, base_time, body["header"]["resultMsg"])
            return rows
        b = body["body"]
        rows += b["items"]["item"]
        if page * 1000 >= b["totalCount"]:
            return rows
        page += 1


def main():
    out_dir = RAW / "kma_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    new = 0
    for base_date, base_time in issued_bases(datetime.now(KST)):
        for name, s in SITES.items():
            path = out_dir / f"{name}_{base_date}_{base_time}.csv"
            if path.exists():
                continue
            rows = fetch(s["nx"], s["ny"], base_date, base_time)
            if rows:
                pd.DataFrame(rows).to_csv(path, index=False)
                new += 1
                print("saved", path.name, len(rows), "rows")
            time.sleep(0.2)
    print("new files:", new)


if __name__ == "__main__":
    main()
