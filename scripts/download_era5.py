"""ERA5 시간자료(입력 기상장)를 Open-Meteo archive API로 다운로드. 키 불필요.

usage: python scripts/download_era5.py 2018-01-01 2025-12-31
"""
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW, SITES  # noqa: E402

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
    "pressure_msl", "precipitation", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation",
]  # 기압면 변수(850hPa 등)는 archive 엔드포인트에서 비어 있어 제외


def get(params):
    for wait in (0, 30, 60, 120, 300):  # 429(요청 한도) 시 대기 후 재시도
        time.sleep(wait)
        r = requests.get(URL, timeout=120, params=params)
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()["hourly"]
    raise RuntimeError("Open-Meteo rate limit: 잠시 후 다시 실행")


def main(start, end):
    out_dir = RAW / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = int(start[:4]), int(end[:4])
    for name, s in SITES.items():
        frames = []
        for y in range(y0, y1 + 1):  # 연 단위 요청
            a = max(start, f"{y}-01-01")
            b = min(end, f"{y}-12-31")
            frames.append(pd.DataFrame(get({
                "latitude": s["lat"], "longitude": s["lon"], "start_date": a, "end_date": b,
                "hourly": ",".join(VARS), "timezone": "Asia/Seoul", "models": "era5",
            })))
            print(name, y)
            time.sleep(5)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(name, len(df), "rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
