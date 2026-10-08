import numpy as np


def rain_scores(y_true_mm, y_pred_rain, threshold_mm=1.0):
    """媛뺤닔 ?댁쭊 遺꾨쪟 吏?? y_pred_rain ? bool(湲곗긽泥??덈낫 ?먮뒗 ?뺣쪧 而룹삤???곸슜 寃곌낵)."""
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
    """寃利?援ш컙?먯꽌 F1 理쒕? 而룹삤?꾨? 李얜뒗?? ?뚯뒪??援ш컙?먮뒗 ??媛믪쓣 怨좎젙 ?곸슜."""
    cuts = np.linspace(0.05, 0.95, 91)
    f1s = [rain_scores(y_true_mm, np.asarray(prob) >= c, threshold_mm)["f1"] for c in cuts]
    return float(cuts[int(np.argmax(f1s))])


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
