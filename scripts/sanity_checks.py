"""Is the within-session 0.71 (xrun) a leak? Controls on the Riemannian baseline.

 1. shuffled labels   : permute y of the TRAIN set only -> must fall to ~0.333 if there is no leak
 2. reversed xrun     : train runs 3-6, test runs 1-2 (does direction matter?)
 3. far apart xrun    : train runs 1-2, test runs 5-6 (more time between train and test)
 4. no-hyperparameter-search note: C, band, estimator were fixed a priori, never tuned on test
Reports mean over 10 subjects of per-subject balanced accuracy, plus the std across subjects.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score as bal
import eval_baselines as E

sub, run, y, X = E.sub, E.run, E.y, E.X
rng = np.random.RandomState(0)


def go(train_runs, test_runs, shuffle=False):
    out = []
    for s in sorted(set(sub)):
        tr = (sub == s) & np.isin(run, train_runs)
        te = (sub == s) & np.isin(run, test_runs)
        ytr = rng.permutation(y[tr]) if shuffle else y[tr]
        out.append(bal(y[te], E.fit_predict_riem(X[tr], ytr, X[te])))
    return np.mean(out), np.std(out), len(out)


for name, args in {
    "baseline xrun (train 1-4, test 5-6)": ([1, 2, 3, 4], [5, 6], False),
    "shuffled train labels (control)": ([1, 2, 3, 4], [5, 6], True),
    "reversed (train 3-6, test 1-2)": ([3, 4, 5, 6], [1, 2], False),
    "far apart (train 1-2, test 5-6)": ([1, 2], [5, 6], False),
}.items():
    m, sd, n = go(*args)
    print(f"{name:40s} mean BA {m:.3f}   std across subjects {sd:.3f}")
