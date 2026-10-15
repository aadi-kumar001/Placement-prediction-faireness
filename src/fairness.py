"""Fairness metrics + mitigation (no external fairness library needed)."""
import numpy as np
import pandas as pd


def group_rates(y_true, y_pred, groups) -> pd.DataFrame:
    rows = []
    for g in np.unique(groups):
        m = groups == g
        pos = y_true[m] == 1
        rows.append(dict(group=g, n=int(m.sum()),
                         selection_rate=y_pred[m].mean(),
                         tpr=y_pred[m][pos].mean() if pos.any() else np.nan,
                         fpr=y_pred[m][~pos].mean() if (~pos).any() else np.nan,
                         base_rate=y_true[m].mean()))
    return pd.DataFrame(rows).set_index("group")


def fairness_report(y_true, y_pred, groups) -> dict:
    r = group_rates(np.asarray(y_true), np.asarray(y_pred), np.asarray(groups))
    sr, tpr, fpr = r.selection_rate, r.tpr, r.fpr
    return dict(
        demographic_parity_diff=float(sr.max() - sr.min()),
        disparate_impact=float(sr.min() / sr.max()) if sr.max() > 0 else np.nan,   # 4/5ths rule: >= 0.8
        equal_opportunity_diff=float(tpr.max() - tpr.min()),
        equalized_odds_diff=float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min())),
        table=r)


def reweighing_weights(y, groups) -> np.ndarray:
    """Kamiran & Calders (2012): w(g,y) = P(g)P(y) / P(g,y). Makes group and label independent."""
    y, groups = np.asarray(y), np.asarray(groups)
    w = np.ones(len(y))
    for g in np.unique(groups):
        for lab in (0, 1):
            m = (groups == g) & (y == lab)
            if m.any():
                w[m] = (groups == g).mean() * (y == lab).mean() / m.mean()
    return w


def group_thresholds(scores, groups, global_rate):
    """Post-processing: per-group threshold so every group gets the same selection rate."""
    scores, groups = np.asarray(scores), np.asarray(groups)
    return {g: float(np.quantile(scores[groups == g], 1 - global_rate)) for g in np.unique(groups)}


def apply_thresholds(scores, groups, th) -> np.ndarray:
    return np.array([int(s >= th[g]) for s, g in zip(scores, groups)])
