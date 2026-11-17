"""Causal (streamable) per-session Euclidean alignment.

Trial i of a session is whitened with R_i^{-1/2}, where R_i only uses windows 0..i of that session (windows already seen):
    R_i = (n0 * scale * G + sum_{j<=i} C_j) / (n0 + i + 1)
C_j = 4-40 Hz spatial covariance of window j (no labels), G = average covariance over all windows, rescaled to the running
mean trace so the prior only supplies the *shape* early on; n0 = prior weight in windows (n0=0: pure running mean).
Compare with the offline version in align_check.py (R from the whole session, which a stream cannot know).
Usage: python scripts/stream_align.py <protocols> <models> <name> <n0,n0,...>
"""
import json, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, "scripts")
import eval_baselines as E


def stream_align(X, cell, run, onset, n0=0.0, band=(4, 40)):
    sos = butter(4, band, btype="band", fs=120, output="sos")
    Xb = sosfiltfilt(sos, X, axis=-1)   # filters each 4 s window on its own samples only (nothing from other windows); check how the official stream delivers windows
    C = np.einsum("nct,ndt->ncd", Xb, Xb) / Xb.shape[-1]
    G = C.mean(0)
    out = np.empty_like(X)
    for c in sorted(set(cell)):
        idx = np.where(cell == c)[0]
        idx = idx[np.lexsort((onset[idx], run[idx]))]            # chronological order inside the session
        cum = np.cumsum(C[idx], axis=0)
        for i, j in enumerate(idx):
            run_mean_tr = np.trace(cum[i]) / (i + 1)
            R = (n0 * G * run_mean_tr / np.trace(G) + cum[i]) / (n0 + i + 1)
            w, v = np.linalg.eigh(R)
            out[j] = (v / np.sqrt(w + 1e-6 * w.max())) @ v.T @ X[j]
    return out.astype(np.float32)


if __name__ == "__main__":
    protocols, models, name = sys.argv[1].split(","), sys.argv[2].split(","), sys.argv[3]
    res = {}
    X0 = E.X
    for n0 in [float(v) for v in sys.argv[4].split(",")]:
        E.X = stream_align(X0, E.cell, E.run, E.D["onset"], n0=n0)
        for p in protocols:
            for m in models:
                res[f"n0={n0:g}/{m}/{p}"] = E.evaluate(m, p)
                print(f"== STREAM n0={n0:g} {m}/{p}: mean BA {res[f'n0={n0:g}/{m}/{p}']['MEAN']:.3f}", flush=True)
    json.dump(res, open(f"results/stream_{name}.json", "w"), indent=1)
