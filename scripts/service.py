"""예보 서비스: 매일 05시 발표분을 우리 앙상블과 기상청 단기예보로 내고, 지난 7일 예보를 관측과 맞춰 본다.

앙상블 = LightGBM 2 : 지점 헤드 5c_cnn 시드 0·1 각 1 (val 로 고른 조합), 강수는 확률 >= CUT 이면 1mm 이상 비.
예보 저장: data/service/fc/{발표 YYYYMMDDHH}.parquet (지점 x 리드, 우리 o_* / 기상청 k_*)
화면 자료: data/service/site/latest.json (오늘 예보 + 지난 7일 성적), verify/{발표}.json (발표별 시간 자료)

usage: python scripts/service.py forecast 2026101605     발표분 예보 (Aurora 박스가 data/aurora/box 에 있어야 함)
       python scripts/service.py seed-test                test 기간 예측(이미 한 번 연 것)을 예보 저장소에 넣기
       python scripts/service.py rain01 2026101305 ...    저장된 발표분에 약한 비(0.1mm) 확률 덧붙이기
       python scripts/service.py site [--now 2026101614]  화면 자료 만들기
       python scripts/service.py archive                  서비스 종료 후 기록: 운영 기간 발표 전부 채점 -> archive.json, verify/
       python scripts/service.py upload                   블로그 Worker 로 올리기 (HSL_Blog/.env 의 PUBLISH_TOKEN)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import RAIN_THRESHOLD_MM, RAW, ROOT, http_get  # noqa: E402
from src.dataset import BOX_DIR, load_asos, load_box, load_meta, model_orography, run_rows, split  # noqa: E402
from src.evaluate import BASE, CLIP  # noqa: E402
from src.station_names import EN  # noqa: E402

RUNS = ROOT / "data" / "train" / "runs"
SVC = ROOT / "data" / "service"
FC, SITE = SVC / "fc", SVC / "site"
MEMBERS = {"1_lgbm_s0": 2, "5c_cnn_s0": 1, "5c_cnn_s1": 1}
CUT = 0.55  # val 에서 정한 앙상블 강수 컷오프 (scripts/ensemble.py)
# 약한 비(0.1mm 이상) 보조 지표: LightGBM 하나 (scripts/train_rain01.py). 1mm 주 지표와 따로 채점한다
RAIN01_MM = 0.1
CUT01 = json.loads((RUNS / "1_lgbm_rn01.json").read_text(encoding="utf-8"))["cut"] if (RUNS / "1_lgbm_rn01.json").exists() else None
VARS = tuple(BASE)  # ta, hm, ws
BLOG = Path(os.environ.get("HSL_BLOG", ROOT.parent / "HSL_Blog"))
SITE_URL = os.environ.get("WEATHER_UPLOAD_URL", "https://hslab.space/api/weather")
LIVE_START = pd.Timestamp("2026-10-13 05:00")  # 실제 서비스로 낸 첫 발표. 그 전(~10/12)은 test 기간 기록


# ---------------------------------------------------------------- 예측
def find_box(base):
    for p in sorted(BOX_DIR.glob("*.npz"), reverse=True):
        with np.load(p) as b:
            if pd.Timestamp(b["base_kst"][()]) == base:
                return p
    raise FileNotFoundError(f"{base:%Y-%m-%d %H시} 발표용 Aurora 박스가 없다 (scripts/aurora_generate.py 로 생성)")


def predict_lgbm(df):
    import lightgbm as lgb
    from train_lgbm import features
    x = features(df)
    load = lambda k: lgb.Booster(model_str=(RUNS / f"1_lgbm_s0_{k}.txt").read_text(encoding="utf-8"))  # noqa: E731
    out = [np.clip(df[BASE[k]].to_numpy() + load(k).predict(x), *CLIP[k]) for k in VARS]
    out.append(load("rn").predict(x))
    return np.stack(out, -1)


def predict_rain01(df):
    import lightgbm as lgb
    from train_lgbm import features
    return lgb.Booster(model_str=(RUNS / "1_lgbm_s0_rn01.txt").read_text(encoding="utf-8")).predict(features(df))


def build_rows(base, meta):
    obs = {s: load_asos(s) for s in meta.stn}
    return run_rows(load_box(find_box(base)), meta, obs, model_orography())


def predict_head(df, tag):
    import torch
    from src.station_model import StationHead
    from train_axial import Grid, physical, predict
    ck = torch.load(RUNS / f"{tag}.pt", map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    meta = load_meta().set_index("stn").loc[ck["stations"]].reset_index()
    stats = {k: ck[k] for k in ("mean", "std", "ystd")} | {"meta": meta}
    if a.grid:
        stats["box_mean"], stats["box_std"] = ck["box_mean"], ck["box_std"]
    model = StationHead(len(ck["cols"]), 5, len(ck["stations"]), d=a.d, depth=a.depth, dropout=a.dropout,
                        stn_attn=not a.no_stn_attn, lead_attn=not a.no_lead_attn, k_ens=a.k, grid=a.grid,
                        n_box_var=len(stats["box_mean"]) if a.grid else 0, obs_ch=5 if a.obs else 0)
    model.load_state_dict(ck["model"])
    return physical(df, predict(model, Grid(df, ck["stations"], ck["cols"], stats, a), "cpu"), ck["ystd"])


def ensemble(parts):
    return np.average(np.stack([parts[t] for t in MEMBERS]), axis=0, weights=list(MEMBERS.values()))


def store(df, p, extra=None):
    """(지점, 리드) 행 + 앙상블 예측 (N, 4) -> 저장 표."""
    out = df[["base_kst", "stn", "lead", "valid_kst"]].copy()
    for j, k in enumerate(VARS):
        out[f"o_{k}"] = p[:, j]
    out["o_pop"] = p[:, -1]
    out["a_tcc"] = df["a_tcc"].to_numpy()  # Aurora 구름량 (0~1), 화면의 하늘 아이콘용
    for k in (*VARS, "rn"):
        out[f"k_{k}"] = df.get(f"k_{k}", np.nan)
    if extra is not None:
        out = out.join(extra, on=["stn", "valid_kst"])
    for c in ("k_pop", "k_sky", "k_pty"):
        if c not in out:
            out[c] = np.nan
    return out


def kma_extra(base):
    p = RAW / "kma_archive" / "extra" / f"{base:%Y%m%d_%H%M}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["valid_kst"]).pivot_table(index=["stn", "valid_kst"], columns="var", values="value")
    return d.rename(columns={"POP": "k_pop", "SKY": "k_sky", "PTY": "k_pty"})


def cmd_forecast(args):
    from src import live
    base = pd.Timestamp(datetime.strptime(args.base, "%Y%m%d%H"))
    meta = load_meta()
    print(f"[service] {base:%Y-%m-%d %H시} 발표 예보", flush=True)
    live.kma_forecast(base, meta)
    # 발표 1시간 전까지 관측: 관측소 파일에 없는 최근 시각만 초단기실황으로 (공식 ASOS 는 전날까지만 나온다)
    last = min(pd.read_csv(RAW / "asos" / "stn" / f"{s}.csv", usecols=["tm"]).tm.max() for s in meta.stn)
    start = max(pd.Timestamp(last) + timedelta(hours=1), base - timedelta(hours=23))
    if start <= base - timedelta(hours=1):
        print(f"[service] 초단기실황으로 관측 채움 {start:%m-%d %H시} ~ {base - timedelta(hours=1):%H시}", flush=True)
        print(f"[service] {live.merge_obs(live.recent_obs(meta, start, base - timedelta(hours=1)))} 행 추가", flush=True)
    df = build_rows(base, meta)
    parts = {"1_lgbm_s0": predict_lgbm(df)} | {t: predict_head(df, t) for t in MEMBERS if t != "1_lgbm_s0"}
    out = store(df, ensemble(parts), kma_extra(base))
    out["o_pop01"] = predict_rain01(df)
    FC.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FC / f"{base:%Y%m%d%H}.parquet", index=False)
    print(f"[service] 저장 {len(out)}행, 비 예보 시간 {(out.o_pop >= CUT).sum()} (기상청 {(out.k_rn >= 1).sum()}), "
          f"약한 비 {(out.o_pop01 >= CUT01).sum()} (기상청 {(out.k_rn >= RAIN01_MM).sum()})")


def cmd_rain01(args):
    """이미 저장한 발표분에 약한 비 확률을 덧붙인다 (박스가 남아 있는 발표만). 기존 열은 건드리지 않는다."""
    meta = load_meta()
    for b in args.bases:
        base = pd.Timestamp(datetime.strptime(b, "%Y%m%d%H"))
        path = FC / f"{base:%Y%m%d%H}.parquet"
        fc = pd.read_parquet(path)
        df = build_rows(base, meta)
        p = pd.Series(predict_rain01(df), index=pd.MultiIndex.from_frame(df[["stn", "valid_kst"]]), name="o_pop01")
        fc = fc.drop(columns=["o_pop01"], errors="ignore").join(p, on=["stn", "valid_kst"])
        fc.to_parquet(path, index=False)
        print(f"[service] {b}: 약한 비 확률 {fc.o_pop01.notna().sum()}/{len(fc)}행")


def cmd_seed_test(_):
    """test 기간(2026-07~10-12) 예측은 이미 한 번 열어 둔 것이라 지난 기록으로 그대로 쓴다."""
    _, _, te = split(pd.read_parquet(ROOT / "data" / "train" / "table.parquet"))
    te = te.reset_index(drop=True)
    p = ensemble({t: np.load(RUNS / f"{t}_test.npy") for t in MEMBERS})
    FC.mkdir(parents=True, exist_ok=True)
    out = store(te, p)
    for b, g in out.groupby("base_kst"):
        g.to_parquet(FC / f"{pd.Timestamp(b):%Y%m%d%H}.parquet", index=False)
    print(f"[service] test 발표 {out.base_kst.nunique()}개 저장")


# ---------------------------------------------------------------- 화면 자료
def r1(a, nd=1):
    """JSON 용: 반올림, nan -> None."""
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(a, dtype=float)]


def with_obs(fc, obs):
    for k in ("ta", "hm", "ws", "rn"):
        fc[f"y_{k}"] = [obs[s][k].get(t, np.nan) if s in obs else np.nan for s, t in zip(fc.stn, fc.valid_kst)]
    return fc


def scores(d):
    """우리·기상청 같은 행에서: 기온·습도·풍속 MAE, 강수 F1 (1mm 이상)."""
    out = {}
    for who, pre in (("ours", "o_"), ("kma", "k_")):
        s = {}
        for k in VARS:
            m = d[f"y_{k}"].notna() & d[f"{pre}{k}"].notna()
            s[k] = round(float((d[f"{pre}{k}"][m] - d[f"y_{k}"][m]).abs().mean()), 3) if m.any() else None
        m = d.y_rn.notna() & (d.o_pop.notna() if who == "ours" else d.k_rn.notna())
        obs = d.y_rn[m] >= RAIN_THRESHOLD_MM
        pred = d.o_pop[m] >= CUT if who == "ours" else d.k_rn[m] >= RAIN_THRESHOLD_MM
        tp, fp, fn = int((obs & pred).sum()), int((~obs & pred).sum()), int((obs & ~pred).sum())
        s["f1"] = round(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else None
        s.update(hits=tp, false=fp, miss=fn)
        # 약한 비(0.1mm 이상): 우리 약한 비 확률이 있는 행만, 기상청도 같은 행으로
        m = d.y_rn.notna() & d.k_rn.notna() & (d["o_pop01"].notna() if "o_pop01" in d else False)
        if CUT01 is not None and m.any():
            obs = d.y_rn[m] >= RAIN01_MM
            pred = d.o_pop01[m] >= CUT01 if who == "ours" else d.k_rn[m] >= RAIN01_MM
            tp, fp, fn = int((obs & pred).sum()), int((~obs & pred).sum()), int((obs & ~pred).sum())
            s["f1_01"] = round(2 * tp / (2 * tp + fp + fn), 3) if tp + fp + fn else None
        out[who] = s
    out["n"] = int(d.y_ta.notna().sum())
    out["rain_hours"] = int((d.y_rn >= RAIN_THRESHOLD_MM).sum())
    return out


def series(g):
    """한 지점 한 발표 -> 화면용 시간 배열."""
    return {
        "ours": {"ta": r1(g.o_ta), "hm": r1(g.o_hm, 0), "ws": r1(g.o_ws), "pop": r1(g.o_pop * 100, 0),
                 "pop01": r1(g.o_pop01 * 100, 0) if "o_pop01" in g else [None] * len(g),
                 "cc": r1(g.a_tcc * 100, 0) if "a_tcc" in g else [None] * len(g)},
        "kma": {"ta": r1(g.k_ta), "hm": r1(g.k_hm, 0), "ws": r1(g.k_ws), "rn": r1(g.k_rn), "pop": r1(g.k_pop, 0),
                "sky": r1(g.k_sky, 0), "pty": r1(g.k_pty, 0)},
        "obs": {"ta": r1(g.y_ta), "hm": r1(g.y_hm, 0), "ws": r1(g.y_ws), "rn": r1(g.y_rn)},
    }


def run_json(fc):
    times = sorted(fc.valid_kst.unique())
    return {"base": f"{pd.Timestamp(fc.base_kst.iloc[0]):%Y-%m-%dT%H:%M}",
            "times": [f"{pd.Timestamp(t):%Y-%m-%dT%H:%M}" for t in times],
            "stations": {str(s): series(g.set_index("valid_kst").reindex(times)) for s, g in fc.groupby("stn")}}


def cmd_site(args):
    now = pd.Timestamp(datetime.strptime(args.now, "%Y%m%d%H")) if args.now else pd.Timestamp.now().floor("h")
    meta = load_meta()
    obs = {s: load_asos(s) for s in meta.stn if (RAW / "asos" / "stn" / f"{s}.csv").exists()}
    files = sorted(FC.glob("*.parquet"))
    bases = [pd.Timestamp(datetime.strptime(p.stem, "%Y%m%d%H")) for p in files]
    latest = max(b for b in bases if b <= now)
    week = [b for b in bases if now - timedelta(days=7) <= b < now - timedelta(hours=1)]
    (SITE / "verify").mkdir(parents=True, exist_ok=True)

    runs, both = [], []
    for b in week:
        fc = with_obs(pd.read_parquet(FC / f"{b:%Y%m%d%H}.parquet"), obs)
        done = fc[fc.y_ta.notna() & fc.k_ta.notna()]
        if done.empty:
            continue
        runs.append({"base": f"{b:%Y-%m-%dT%H:%M}", **scores(done)})
        both.append(done)
        (SITE / "verify" / f"{b:%Y%m%d%H}.json").write_text(json.dumps(run_json(fc), separators=(",", ":")))
    week_df = pd.concat(both, ignore_index=True) if both else None

    fc = with_obs(pd.read_parquet(FC / f"{latest:%Y%m%d%H}.parquet"), obs)
    site = {
        "updated": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M}",
        "cut": CUT,
        "cut01": CUT01,
        "stations": [{"stn": int(s.stn), "name": s.name, "nameEn": EN.get(s.stn, s.name), "lat": round(s.lat, 4),
                      "lon": round(s.lon, 4)} for s in meta.itertuples()],
        "forecast": run_json(fc),
        "week": None if week_df is None else {
            "from": runs[0]["base"], "to": runs[-1]["base"], "runs": runs, "total": scores(week_df),
            "stations": {str(s): scores(g) for s, g in week_df.groupby("stn")},
        },
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "latest.json").write_text(json.dumps(site, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[service] 화면 자료: 예보 {latest:%m-%d %H시}, 지난 7일 발표 {len(runs)}개")
    if week_df is not None:
        t = site["week"]["total"]
        print("  우리  ", t["ours"], "\n  기상청", t["kma"])


def cmd_archive(_):
    """서비스 종료 후 기록: 운영 기간(LIVE_START~) 발표 전부를 지금까지의 관측으로 채점한다. 예측값은 저장된 그대로 쓴다."""
    meta = load_meta()
    obs = {s: load_asos(s) for s in meta.stn if (RAW / "asos" / "stn" / f"{s}.csv").exists()}
    bases = sorted(pd.Timestamp(datetime.strptime(p.stem, "%Y%m%d%H")) for p in FC.glob("*.parquet"))
    (SITE / "verify").mkdir(parents=True, exist_ok=True)
    runs, both = [], []
    for b in (x for x in bases if x >= LIVE_START):
        fc = with_obs(pd.read_parquet(FC / f"{b:%Y%m%d%H}.parquet"), obs)
        done = fc[fc.y_ta.notna() & fc.k_ta.notna()]
        (SITE / "verify" / f"{b:%Y%m%d%H}.json").write_text(json.dumps(run_json(fc), separators=(",", ":")))
        if done.empty:
            continue
        runs.append({"base": f"{b:%Y-%m-%dT%H:%M}", "partial": bool(done.valid_kst.max() < fc.valid_kst.max()), **scores(done)})
        both.append(done)
    df = pd.concat(both, ignore_index=True)
    better = lambda r, k: None if r["ours"].get(k) is None or r["kma"].get(k) is None else (  # noqa: E731
        r["ours"][k] < r["kma"][k] if k in VARS else r["ours"][k] > r["kma"][k])
    wins = {k: [sum(better(r, k) is True for r in runs), sum(better(r, k) is not None for r in runs)] for k in (*VARS, "f1", "f1_01")}
    site = {
        "updated": f"{pd.Timestamp.now():%Y-%m-%dT%H:%M}",
        "cut": CUT,
        "cut01": CUT01,
        "stations": [{"stn": int(s.stn), "name": s.name, "nameEn": EN.get(s.stn, s.name), "lat": round(s.lat, 4),
                      "lon": round(s.lon, 4)} for s in meta.itertuples()],
        "live": {"from": runs[0]["base"], "to": runs[-1]["base"], "runs": runs, "total": scores(df), "wins": wins,
                 "stations": {str(s): scores(g) for s, g in df.groupby("stn")}},
    }
    (SITE / "archive.json").write_text(json.dumps(site, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    t = site["live"]["total"]
    print(f"[service] 기록: 발표 {len(runs)}개, 일부만 채점 {[r['base'] for r in runs if r['partial']]}")
    print("  우리  ", t["ours"], "\n  기상청", t["kma"], "\n  시간", t["n"], "비 온 시간", t["rain_hours"], "\n  이긴 발표", wins)


def cmd_upload(_):
    from src.config import load_env
    for line in (BLOG / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("PUBLISH_TOKEN="):
            os.environ.setdefault("PUBLISH_TOKEN", line.split("=", 1)[1].strip().strip('"'))
    load_env()
    import requests
    token = os.environ["PUBLISH_TOKEN"]
    files = [SITE / "latest.json", *[SITE / "archive.json"] * (SITE / "archive.json").exists(), *sorted((SITE / "verify").glob("*.json"))]
    for f in files:
        name = f.relative_to(SITE).as_posix()
        r = requests.put(f"{SITE_URL}/{name}", data=f.read_bytes(), timeout=60,
                         headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        if r.status_code >= 300:
            raise RuntimeError(f"올리기 실패 {name}: {r.status_code} {r.text[:200]}")
    print(f"[service] {len(files)}개 올림 -> {SITE_URL}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("forecast").add_argument("base", help="발표시각 YYYYMMDDHH (KST)")
    sub.add_parser("seed-test")
    sub.add_parser("rain01").add_argument("bases", nargs="+", help="약한 비 확률을 덧붙일 발표시각 YYYYMMDDHH")
    sub.add_parser("site").add_argument("--now", help="기준 시각 YYYYMMDDHH (기본 지금)")
    sub.add_parser("archive")
    sub.add_parser("upload")
    args = ap.parse_args()
    {"forecast": cmd_forecast, "seed-test": cmd_seed_test, "rain01": cmd_rain01, "site": cmd_site,
     "archive": cmd_archive, "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
