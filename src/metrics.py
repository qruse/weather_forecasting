import numpy as np


def rain_scores(y_true_mm, y_pred_rain, threshold_mm=1.0):
    """강수 이진 분류 지표. y_pred_rain 은 bool(기상청 예보 또는 확률 컷오프 적용 결과)."""
    t = np.asarray(y_true_mm) >= threshold_mm
    p = np.asarray(y_pred_rain).astype(bool)
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "precision": prec,
        "recall": rec,
        "ts": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "acc": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def best_cutoff(y_true_mm, prob, threshold_mm=1.0):
    """검증 구간에서 F1 최대 컷오프를 찾는다. 테스트 구간에는 이 값을 고정 적용."""
    cuts = np.linspace(0.05, 0.95, 91)
    f1s = [rain_scores(y_true_mm, np.asarray(prob) >= c, threshold_mm)["f1"] for c in cuts]
    return float(cuts[int(np.argmax(f1s))])


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
