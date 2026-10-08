"""Aurora 1.5 입력(Batch) 생성과 지점 추출.

입력: ARCO-ERA5 (gs://gcp-public-data-arco-era5, 0.25도, 시간 단위, 인증 불필요)
정적 변수: HF microsoft/aurora 의 aurora-0.25-v1.5-static.pickle
"""
import pickle
from datetime import datetime, timedelta

import numpy as np
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

from aurora import Batch, Metadata
from aurora.insolation import insolation

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

# Aurora 변수명 -> ARCO-ERA5 변수명 (출력 전용 변수는 입력에 넣지 않는다)
SURF = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "2d": "2m_dewpoint_temperature",
    "tcwv": "total_column_water_vapour",
    "tcc": "total_cloud_cover",
    "100u": "100m_u_component_of_wind",
    "100v": "100m_v_component_of_wind",
    "sp": "surface_pressure",
    "lcc": "low_cloud_cover",
    "mcc": "medium_cloud_cover",
    "hcc": "high_cloud_cover",
    "skt": "skin_temperature",
    "stl1": "soil_temperature_level_1",
    "swvl1": "volumetric_soil_water_layer_1",
    "ci": "sea_ice_cover",
    "scaled_sd": "snow_depth",  # 이름과 달리 스케일하지 않은 값을 넣는다(문서)
}
ATMOS = {
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "t": "temperature",
    "q": "specific_humidity",
}

_ds = None


def open_arco():
    global _ds
    if _ds is None:
        _ds = xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})
    return _ds


def load_static():
    path = hf_hub_download("microsoft/aurora", "aurora-0.25-v1.5-static.pickle")
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in raw.items()}


def build_batch(init_time: datetime, static=None, history_hours=6, with_insolation=True) -> Batch:
    """init_time(UTC, naive) 시점의 분석장과 history_hours 전 분석장으로 Batch 를 만든다."""
    ds = open_arco()
    times = [init_time - timedelta(hours=history_hours), init_time]
    tsel = np.array(times, dtype="datetime64[ns]")
    static = static or load_static()

    surf_ds = ds[list(SURF.values())].sel(time=tsel).load()
    atm_ds = ds[list(ATMOS.values())].sel(time=tsel, level=list(LEVELS)).load()

    lat = surf_ds.latitude.values.astype(np.float32)  # 90 -> -90
    lon = surf_ds.longitude.values.astype(np.float32)  # 0 -> 359.75

    # 해빙(ci)·토양 변수는 육지/해양 마스크 밖이 NaN 이라 0 으로 채운다
    surf = {
        k: torch.from_numpy(np.nan_to_num(surf_ds[v].values.astype(np.float32)))[None]
        for k, v in SURF.items()
    }
    atmos = {k: torch.from_numpy(atm_ds[v].values.astype(np.float32))[None] for k, v in ATMOS.items()}

    if with_insolation:
        sol = np.stack([insolation([t], lat, lon, enforce_2d=True)[0] for t in times])
        surf["insolation"] = torch.from_numpy(sol.astype(np.float32))[None]

    return Batch(
        surf_vars=surf,
        static_vars=static,
        atmos_vars=atmos,
        metadata=Metadata(
            lat=torch.from_numpy(lat),
            lon=torch.from_numpy(lon),
            time=(init_time,),
            atmos_levels=LEVELS,
        ),
    )


def bilinear(field, lat, lon):
    """(H, W) 격자(위도 90→-90, 경도 0→359.75)에서 한 점의 쌍선형 보간값."""
    h, w = field.shape
    y = (90.0 - lat) / 0.25
    x = (lon % 360.0) / 0.25
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, h - 1), (x0 + 1) % w
    fy, fx = y - y0, x - x0
    f = np.asarray(field)
    return float(
        f[y0, x0] * (1 - fy) * (1 - fx) + f[y0, x1] * (1 - fy) * fx
        + f[y1, x0] * fy * (1 - fx) + f[y1, x1] * fy * fx
    )


def rel_humidity(t_c, td_c):
    """이슬점으로 상대습도(%) 계산(Magnus)."""
    a, b = 17.625, 243.04
    return float(100 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c)))


def station_values(pred: Batch, lat, lon):
    """예측 Batch 에서 한 지점의 기온(℃), 습도(%), 풍속(m/s), 강수(모델 단위) 를 뽑는다."""
    g = lambda k: bilinear(pred.surf_vars[k][0, 0].cpu().numpy(), lat, lon)  # noqa: E731
    t, td = g("2t") - 273.15, g("2d") - 273.15
    out = {
        "ta": t,
        "hm": rel_humidity(t, td),
        "ws": float(np.hypot(g("10u"), g("10v"))),
    }
    if "scaled_tp_1h" in pred.surf_vars:
        out["tp_raw"] = g("scaled_tp_1h")
    return out
