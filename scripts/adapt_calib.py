"""Using labelled calibration data of the target subject (Riemannian + causal alignment).

Two scenarios, both forward in time (calibration is always EARLIER than the test data, like the competition):
  fwd_sess : P04/P10 only. Test = session 2 or 3; calibration = the subject's earlier sessions (4 test cells).
  calib_run: all 10 subjects. Test = runs 5-6 of the session; calibration = runs 1-4 of the same session (proxy; 10 cells).
Training sets compared (all other subjects' data is always available):
  own_only   : calibration data only (per-subject model)
  others_only: other subjects only (zero-shot, = no calibration)
  pooled     : others + calibration, equal weight per trial
  w_own=K    : others + calibration, each calibration trial weighted K times
Usage: python scripts/adapt_calib.py <name>
"""
import json, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as bal
from sklearn.pipeline import make_pipeline
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, "scripts")
import eval_baselines as E
from stream_align import stream_align

X = stream_align(E.X, E.cell, E.run, E.D["onset"], n0=0.0)
y, sub, ses, run = E.y, E.sub, E.ses, E.run
SOS = butter(4, [8, 30], btype="band", fs=120, output="sos")
XF = sosfiltfilt(SOS, X, axis=-1)


def fit_predict(tr, te, w):
    pipe = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"),
                         LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    pipe.fit(XF[tr], y[tr], tangentspace__sample_weight=w, logisticregression__sample_weight=w)
    return pipe.predict(XF[te])


def run_all(tr_own, others, te):
    out = {}
    out["own_only"] = bal(y[te], fit_predict(tr_own, te, np.ones(tr_own.sum())))
    out["others_only"] = bal(y[te], fit_predict(others, te, np.ones(others.sum())))
    both = tr_own | others
    for name, k in [("pooled", 1), ("w_own=5", 5), ("w_own=20", 20)]:
        w = np.where(tr_own[both], float(k), 1.0)
        out[name] = bal(y[te], fit_predict(both, te, w))
    return out


def scenario(cases):
    rows = {}
    for name, tr_own, others, te in cases:
        rows[name] = run_all(tr_own, others, te)
        print("  " + name.ljust(18) + "  ".join(f"{k}={v:.3f}" for k, v in rows[name].items()), flush=True)
    keys = list(next(iter(rows.values())))
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}
    print("  MEAN".ljust(20) + "  ".join(f"{k}={v:.3f}" for k, v in mean.items()), flush=True)
    return {"cells": rows, "mean": mean}


res = {}
print("== fwd_sess (calibration = earlier sessions of the same subject)")
cases = []
for s, t in [("sub-P04", "ses-2"), ("sub-P04", "ses-3"), ("sub-P10", "ses-2"), ("sub-P10", "ses-3")]:
    own = (sub == s) & (ses < t)
    cases.append((f"{s}/{t}", own, sub != s, (sub == s) & (ses == t)))
res["fwd_sess"] = scenario(cases)

print("== calib_run (calibration = runs 1-4 of the same session, test = runs 5-6)")
cases = [(s, (sub == s) & (run <= 4), sub != s, (sub == s) & (run >= 5)) for s in sorted(set(sub))]
res["calib_run"] = scenario(cases)
json.dump(res, open(f"results/adapt_{sys.argv[1]}.json", "w"), indent=1)
