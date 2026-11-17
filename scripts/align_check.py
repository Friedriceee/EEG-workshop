"""Does per-session Euclidean alignment (EA) help? OFFLINE upper bound, not yet causal.

For every (subject, session) cell, estimate the mean spatial covariance R of that cell's 4-40 Hz data (no labels used) and
whiten every trial of the cell with R^{-1/2}. The same transform is applied to train and test cells alike.
Offline = R uses all trials of the cell; a streamed submission can only use windows seen so far (running mean).
Usage: EEG_VAL=group python scripts/align_check.py <protocols> <models> <name>
"""
import json, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, "scripts")
import eval_baselines as E


def euclidean_align(X, cell, band=(4, 40)):
    sos = butter(4, band, btype="band", fs=120, output="sos")
    out = np.empty_like(X)
    for c in sorted(set(cell)):
        m = cell == c
        Xb = sosfiltfilt(sos, X[m], axis=-1)
        C = np.einsum("nct,ndt->cd", Xb, Xb) / (Xb.shape[0] * Xb.shape[2])
        w, v = np.linalg.eigh(C)
        R = (v / np.sqrt(w + 1e-6 * w.max())) @ v.T
        out[m] = np.einsum("cd,ndt->nct", R, X[m])
    return out.astype(np.float32)


if __name__ == "__main__":
    E.X = euclidean_align(E.X, E.cell)
    res = {}
    print(f"validation mode: {E.VAL_MODE}  seed: {E.SEED}  (inputs: per-session Euclidean alignment)", flush=True)
    for p in sys.argv[1].split(","):
        for m in sys.argv[2].split(","):
            res[f"{m}/{p}"] = E.evaluate(m, p)
            print(f"== EA {m}/{p}: mean BA {res[f'{m}/{p}']['MEAN']:.3f}", flush=True)
    json.dump(res, open(f"results/align_{sys.argv[3]}.json", "w"), indent=1)
