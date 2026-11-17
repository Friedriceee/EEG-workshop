"""Fit the weights used by submission/submission.py (per-subject models + one generic model).

fit_weights(own_masks, generic_mask) -> dict of numpy arrays/lists (same keys submission.py reads).
Training features are computed with the SAME causal alignment as inference (stream_align, n0=0), then 8-30 Hz,
OAS covariance, tangent space (pyriemann reference) and a class-balanced logistic regression.
Usage: python scripts/train_submission.py            -> writes submission/weights.pt from ALL training data
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt_zi, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, "scripts")
import eval_baselines as E
from stream_align import stream_align

SOS_A = butter(4, [4, 40], btype="band", fs=120, output="sos")
SOS_B = butter(4, [8, 30], btype="band", fs=120, output="sos")
XA = stream_align(E.X, E.cell, E.run, E.D["onset"], n0=0.0)          # causal alignment, as at inference
XF = sosfiltfilt(SOS_B, XA, axis=-1)


def fit_one(mask):
    cov = Covariances(estimator="oas").transform(XF[mask])
    ts = TangentSpace(metric="riemann").fit(cov)
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(ts.transform(cov), E.y[mask])
    return ts.reference_, lr.coef_, lr.intercept_


def fit_weights(own_masks, generic_mask):
    w = {"sos_align": SOS_A, "zi_align": sosfilt_zi(SOS_A), "sos_band": SOS_B, "zi_band": sosfilt_zi(SOS_B),
         "ch_names": [str(c) for c in E.D["ch_names"]]}
    for name, mask in list(own_masks.items()) + [("generic", generic_mask)]:
        w[f"{name}/ref"], w[f"{name}/coef"], w[f"{name}/intercept"] = fit_one(mask)
    return w


if __name__ == "__main__":
    import torch
    subs = sorted(set(E.sub))
    w = fit_weights({s: E.sub == s for s in subs}, np.ones(len(E.y), bool))
    out = {k: (torch.tensor(v) if isinstance(v, np.ndarray) else v) for k, v in w.items()}
    torch.save(out, "submission/weights.pt")
    print("saved submission/weights.pt", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})
