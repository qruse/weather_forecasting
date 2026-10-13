"""WeatherBench 2 공식 결과(gs://weatherbench2/benchmark_results)에서 모델별 점수를 뽑아 비교한다.

usage: python scripts/wb2_compare.py [연도] [정답: hres_t0|era5]
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import pandas as pd
import xarray as xr

YEAR = sys.argv[1] if len(sys.argv) > 1 else "2022"
GT = sys.argv[2] if len(sys.argv) > 2 else "hres_t0"
MODELS = {
    "2022": ["hres", "aurora", "graphcast", "pangu", "fgn_operational_mean",
             "gencast_operational_100m_uv_mean", "climatology", "persistence"],
    "2020": ["hres", "graphcast", "fuxi", "gencast_mean", "pangu", "ens_mean",
             "climatology", "persistence"],
}[YEAR]
VARS = ["2m_temperature", "10m_wind_speed", "mean_sea_level_pressure"]
LEADS = [24, 48, 72, 120, 240]

fs = gcsfs.GCSFileSystem(token="anon")


def load(m):
    path = f"weatherbench2/benchmark_results/{m}_vs_{GT}_240x121_{YEAR}.nc"
    try:
        with fs.open(path) as f:
            return m, xr.load_dataset(f)
    except Exception as e:
        print("missing", m, type(e).__name__, flush=True)
        return m, None


with ThreadPoolExecutor(8) as ex:
    D = {m: d for m, d in ex.map(load, MODELS) if d is not None}
print("loaded:", list(D), flush=True)

d0 = next(iter(D.values()))
regions = [str(r) for r in d0.region.values]
print("regions:", regions, flush=True)
print("precip/seeps vars:", sorted({v for d in D.values() for v in d.data_vars
                                   if "precip" in v or "seeps" in v.lower()}), flush=True)

for region in ("global", "east-asia", "east-asia_land"):
    if region not in regions:
        continue
    for var in VARS:
        rows = {}
        for m, d in D.items():
            k = f"rmse.{var}"
            if k in d:
                lt = d.lead_time
                leads = LEADS if lt.dtype.kind in "iu" else pd.to_timedelta(LEADS, unit="h")
                rows[m] = d[k].sel(region=region).sel(lead_time=leads, method="nearest").values
        print(f"\n== RMSE {var} | region={region} | {YEAR} vs {GT}")
        print(pd.DataFrame(rows, index=[f"{h}h" for h in LEADS]).T.round(3).to_string(), flush=True)
