"""Local dry run of submission/submission.py, streamed the way NeuralBench streams (one window at a time, reset per stream).

1. Equivalence: numpy filter / OAS covariance / tangent space vs scipy / pyriemann on real windows.
2. Forward-in-time scenario with per-subject models trained ONLY on earlier data:
   P04/P10 sessions 2 and 3 (own = earlier sessions) and runs 5-6 of every subject (own = runs 1-4 of that session).
   The stream is fed in (run, onset) order, reset_state() at each new (subject, session).
Expected from the vectorised experiments (scripts/adapt_calib.py, own_only): fwd_sess 0.545, calib_run 0.717.
"""
import sys, time
import numpy as np
from scipy.signal import sosfiltfilt as sp_sosfiltfilt
from sklearn.metrics import balanced_accuracy_score as bal

sys.path.insert(0, "scripts"); sys.path.insert(0, "submission")
import eval_baselines as E
import train_submission as T
import submission as S

X, y, sub, ses, run, onset = E.X, E.y, E.sub, E.ses, E.run, E.D["onset"]

print("== 1. numerical equivalence (numpy vs scipy/pyriemann)")
x = X[100].astype(np.float64)
ref = sp_sosfiltfilt(T.SOS_B, x, axis=-1)
mine = S.sosfiltfilt(T.SOS_B, np.asarray(__import__("scipy.signal", fromlist=["sosfilt_zi"]).sosfilt_zi(T.SOS_B)), x)
print(f"   band-pass max abs diff {np.abs(ref - mine).max():.2e} (signal sd {ref.std():.1f})")
from pyriemann.estimation import Covariances
c_ref = Covariances(estimator="oas").transform(x[None])[0]
print(f"   OAS covariance max rel diff {np.abs(c_ref - S._oas_cov(x)).max() / np.abs(c_ref).max():.2e}")

print("== 2. streamed forward-in-time test")
cases = []
for s, t in [("sub-P04", "ses-2"), ("sub-P04", "ses-3"), ("sub-P10", "ses-2"), ("sub-P10", "ses-3")]:
    cases.append((f"{s}/{t}", "fwd_sess", {s: (sub == s) & (ses < t)}, (sub == s) & (ses == t), s))
for s in sorted(set(sub)):
    cases.append((f"{s}", "calib_run", {s: (sub == s) & (run <= 4)}, (sub == s) & (run >= 5), s))

res = {"fwd_sess": [], "calib_run": []}
for name, scen, own, te, s in cases:
    t0 = time.time()
    solver = S.RiemannianStreamSolver(T.fit_weights(own, ~te & (sub != s)))   # generic = other subjects only
    meta = {"subject": s, "sfreq": 120}
    idx_all, pred_all, cell_ba = [], [], []
    for sess in sorted(set(ses[te])):                      # one stream per (subject, session), state reset at each
        idx = np.where(te & (ses == sess))[0]
        idx = idx[np.lexsort((onset[idx], run[idx]))]
        solver.reset_state()
        # windows of this session that come BEFORE the scored ones (calibration runs) are part of the same stream:
        # replay them without scoring so the running covariance is what it would be online
        warm = np.where((sub == s) & (ses == sess) & ~te)[0]
        warm = warm[np.lexsort((onset[warm], run[warm]))]
        for j in warm:
            solver.predict_proba(X[j], meta)
        pred = [solver.predict(X[j], meta) for j in idx]
        idx_all += list(idx); pred_all += pred; cell_ba.append(bal(y[idx], pred))
    # pooled BA over the subject's test windows (what adapt_calib.py reported) and official-style mean over cells
    res[scen].append((bal(y[idx_all], pred_all), float(np.mean(cell_ba))))
    print(f"   {scen:9s} {name:15s} pooled BA={res[scen][-1][0]:.3f}  per-cell mean={res[scen][-1][1]:.3f}"
          f"  ({time.time() - t0:.0f}s, {len(idx_all)} windows, {len(cell_ba)} stream(s))", flush=True)
for k, v in res.items():
    print(f"== {k}: mean pooled BA {np.mean([a for a, _ in v]):.3f}   mean per-cell BA {np.mean([b for _, b in v]):.3f}")
