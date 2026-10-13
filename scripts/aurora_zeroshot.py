"""Aurora 1.5 사전학습 모델(무파인튜닝)로 서울·대구 시간별 예보를 한 번 돌려 본다.

동아시아 영역만 잘라 넣고 6시간 스텝 + 1시간 서브스텝으로 롤아웃한다.
ASOS 파일이 있으면 같은 시각 관측과 나란히 놓고 MAE 를 찍는다.

usage: python scripts/aurora_zeroshot.py [초기시각 UTC, 예: 2025-07-16T00] [시간, 6의 배수]
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora.rollout import rollout  # noqa: E402

from src.aurora_data import EAST_ASIA, build_batch, station_values  # noqa: E402
from src.aurora_model import load_aurora  # noqa: E402
from src.config import RAIN_THRESHOLD_MM, RAW, ROOT, SITES  # noqa: E402
from src.metrics import mae  # noqa: E402

INIT = datetime.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else datetime(2025, 7, 16, 0)
HOURS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
KST = timedelta(hours=9)

t0 = time.time()
batch = build_batch(INIT, region=EAST_ASIA)
print(f"입력 준비 {time.time() - t0:.0f}s, 격자 {tuple(batch.spatial_shape)}", flush=True)

model = load_aurora(regional=True)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.reset_peak_memory_stats()

t0 = time.time()
rows = []
with torch.inference_mode():
    for pred in rollout(model, batch, steps=HOURS // 6, fine_lead_times=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        valid = pred.metadata.time[0]
        for site, s in SITES.items():
            v = station_values(pred, s["lat"], s["lon"])
            rows.append({
                "site": site,
                "lead_h": int((valid - INIT).total_seconds() // 3600),
                "valid_kst": valid + KST,
                "ta": v["ta"], "hm": v["hm"], "ws": v["ws"],
                # ERA5 강수 단위(m)로 학습된 것으로 보고 mm 로 바꾼다
                "rn": max(v.get("tp_raw", float("nan")), 0.0) * 1000,
            })
print(f"롤아웃 {time.time() - t0:.0f}s", flush=True)
if cuda:
    print(f"GPU {torch.cuda.get_device_name()} 최대 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

fc = pd.DataFrame(rows)
out = ROOT / "data" / "aurora" / f"zeroshot_{INIT:%Y%m%d%H}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fc.to_csv(out, index=False)
print("저장:", out.relative_to(ROOT))

for site in SITES:
    f = fc[fc.site == site].set_index("valid_kst")
    path = RAW / "asos" / f"{site}.csv"
    if path.exists():
        obs = pd.read_csv(path, usecols=["tm", "ta", "hm", "ws", "rn"], parse_dates=["tm"])
        obs = obs.set_index("tm").fillna({"rn": 0.0})
        f = f.join(obs, rsuffix="_obs", how="left")
    print(f"\n== {site} (초기 {INIT + KST:%Y-%m-%d %H시} KST)")
    cols = [c for c in ["lead_h", "ta", "ta_obs", "hm", "hm_obs", "ws", "ws_obs", "rn", "rn_obs"] if c in f]
    print(f[cols].iloc[::3].round(1).to_string())
    if "ta_obs" in f:
        ok = f.dropna(subset=["ta_obs", "hm_obs", "ws_obs"])
        print("MAE 기온 {:.2f}℃ 습도 {:.1f}% 풍속 {:.2f}m/s | 비(≥{}mm) 예측 {}시간 / 관측 {}시간".format(
            mae(ok.ta_obs, ok.ta), mae(ok.hm_obs, ok.hm), mae(ok.ws_obs, ok.ws), RAIN_THRESHOLD_MM,
            int((f.rn >= RAIN_THRESHOLD_MM).sum()), int((f.rn_obs >= RAIN_THRESHOLD_MM).sum()),
        ))
