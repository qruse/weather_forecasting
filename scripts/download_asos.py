"""ASOS 시간자료(관측, 정답) 다운로드. 공공데이터포털 serviceKey 필요.

usage: python scripts/download_asos.py 2018-01-01 2025-12-31
env:   DATA_GO_KR_KEY
"""
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW, SITES  # noqa: E402

URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
KEY = os.environ["DATA_GO_KR_KEY"]


def fetch(stn, start, end):
    rows, page = [], 1
    while True:
        r = requests.get(URL, timeout=60, params={
            "serviceKey": KEY, "pageNo": page, "numOfRows": 999, "dataType": "JSON",
            "dataCd": "ASOS", "dateCd": "HR", "stnIds": stn,
            "startDt": start.strftime("%Y%m%d"), "startHh": "00",
            "endDt": end.strftime("%Y%m%d"), "endHh": "23",
        })
        r.raise_for_status()
        body = r.json()["response"]["body"]
        items = body["items"]["item"] if body.get("items") else []
        rows += items
        if page * 999 >= body["totalCount"]:
            return rows
        page += 1
        time.sleep(0.2)


def main(start, end):
    out_dir = RAW / "asos"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in SITES.items():
        frames, cur = [], start
        while cur <= end:  # 연 단위로 나눠 요청
            nxt = min(date(cur.year, 12, 31), end)
            frames.append(pd.DataFrame(fetch(s["stn"], cur, nxt)))
            print(name, cur, nxt)
            cur = nxt + timedelta(days=1)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        print(name, len(df), "rows")


if __name__ == "__main__":
    main(date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]))
