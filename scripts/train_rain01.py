"""약한 비(0.1mm 이상) 분류기: LightGBM, 서비스의 1mm 앙상블과 따로 둔다 (임시, CPU).

1mm 기준은 대결 시작 때 정한 주 지표라 그대로 두고, 이슬비까지 비로 보는 0.1mm 를 보조 지표로 더한다.
입력·구간·하이퍼파라미터는 train_lgbm.py 와 같다. 컷오프는 val F1 최대, test 는 이 모델로 처음 한 번 연다.
기상청은 단기예보 강수량을 그대로 쓴다 ("1mm 미만" = 0.5 라 0.1mm 기준에서는 비 예보).

usage: python scripts/train_rain01.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import TRAIN  # noqa: E402
from src.dataset import split  # noqa: E402
from src.metrics import best_cutoff, rain_scores  # noqa: E402
from train_lgbm import PARAMS, features  # noqa: E402

TH = 0.1
RUNS = TRAIN / "runs"


def compare(name, d, prob, cut):
    m = d.k_rn.notna().to_numpy()
    y = d.y_rn.to_numpy()
    r4 = lambda s: {k: round(float(v), 4) for k, v in s.items()}  # noqa: E731
    out = {"ours": r4(rain_scores(y[m], prob[m] >= cut, TH)),
           f"kma_{TH}": r4(rain_scores(y[m], d.k_rn.to_numpy()[m] >= TH, TH))}
    out["n"] = int(m.sum())
    out["rain_hours"] = int((y[m] >= TH).sum())
    print(f"[{name}] n={out['n']} 비(0.1mm+) {out['rain_hours']}시간")
    print(f"  우리   F1 {out['ours']['f1']:.3f}  정밀도 {out['ours']['precision']:.3f}  재현율 {out['ours']['recall']:.3f}")
    k = out[f"kma_{TH}"]
    print(f"  기상청 F1 {k['f1']:.3f}  정밀도 {k['precision']:.3f}  재현율 {k['recall']:.3f}")
    return out


def main():
    df = pd.read_parquet(TRAIN / "table.parquet")
    tr, va, te = split(df)
    tr, va, te = (x[x.y_rn.notna()].reset_index(drop=True) for x in (tr, va, te))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}", flush=True)
    xtr, xva, xte = features(tr), features(va), features(te)
    model = lgb.train({**PARAMS, "objective": "binary"}, lgb.Dataset(xtr, (tr.y_rn >= TH).astype(int)), 3000,
                      valid_sets=[lgb.Dataset(xva, (va.y_rn >= TH).astype(int))],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    it = model.best_iteration
    pva, pte = model.predict(xva, num_iteration=it), model.predict(xte, num_iteration=it)
    cut = best_cutoff(va.y_rn, pva, TH)
    print(f"{it} rounds, 컷오프 {cut:.2f}", flush=True)
    result = {"name": "1_lgbm_rn01", "threshold_mm": TH, "rounds": it, "cut": cut,
              "val": compare("val", va, pva, cut), "test": compare("test", te, pte, cut)}
    # LightGBM 파일 저장은 한글 경로를 못 열어서 문자열로 쓴다 (train_lgbm.py 와 같음)
    (RUNS / "1_lgbm_s0_rn01.txt").write_text(model.model_to_string(num_iteration=it), encoding="utf-8")
    (RUNS / "1_lgbm_rn01.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    np.save(RUNS / "1_lgbm_rn01_test.npy", pte.astype(np.float32))


if __name__ == "__main__":
    main()
