from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# stn: ASOS 지점번호, nx/ny: 동네예보 격자
SITES = {
    "seoul": {"stn": 108, "lat": 37.5714, "lon": 126.9658, "nx": 60, "ny": 127},
    "daegu": {"stn": 143, "lat": 35.8850, "lon": 128.6190, "nx": 89, "ny": 90},
}

# 동네예보 발표 시각 (KST)
BASE_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]

RAIN_THRESHOLD_MM = 1.0  # 1mm 이상을 비로 판정
