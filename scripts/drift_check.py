"""Is the large amplitude in many trials slow drift, and does removing it change the baselines?

Part 1: fraction of each trial's variance below 1 Hz; share of |x|>200 uV trials before/after a 1 Hz high-pass.
Part 2: rerun xrun/xsess for Riemannian and EEGNet with the 1 Hz high-pass applied to every window.
(High-pass is applied per 4 s window with a zero-phase filter, so it is not causal; it only tells us whether
drift matters. A causal version would be needed for the streamed submission.)
"""
import sys
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, "scripts")
import eval_baselines as E

X0 = E.X
sos = butter(2, 1.0, btype="high", fs=120, output="sos")
X1 = sosfiltfilt(sos, X0, axis=-1, padlen=60).astype(np.float32)

sub = E.sub
cells = sorted(set(E.cell))
f, P = welch(X0, fs=120, nperseg=480, axis=-1)
low = P[..., f < 1].sum(-1).mean(-1) / P[..., (f >= 0.1) & (f < 45)].sum(-1).mean(-1)

print("== Part 1: drift share and effect of a 1 Hz high-pass")
print(f"{'subject':8s} {'power<1Hz share':>16s} {'trials>200uV raw':>18s} {'after 1Hz HP':>14s} {'median sd raw':>14s} {'after':>8s}")
for s in sorted(set(sub)):
    m = sub == s
    a = (np.abs(X0[m]).max(axis=(1, 2)) > 200).mean(); b = (np.abs(X1[m]).max(axis=(1, 2)) > 200).mean()
    print(f"{s:8s} {np.median(low[m]):16.2f} {a:18.1%} {b:14.1%} {np.median(X0[m].std(-1)):14.1f} {np.median(X1[m].std(-1)):8.1f}")
print(f"all      {np.median(low):16.2f} {(np.abs(X0).max(axis=(1,2))>200).mean():18.1%} {(np.abs(X1).max(axis=(1,2))>200).mean():14.1%}")

print("\n== Part 2: baselines with 1 Hz high-pass (before -> after); noise level ~ +-0.03")
before = {("riem", "xrun"): .714, ("riem", "xsess"): .526, ("eegnet", "xrun"): .425, ("eegnet", "xsess"): .526}
E.X = X1
for model in ["riem", "eegnet"]:
    for proto in ["xrun", "xsess"]:
        r = E.evaluate(model, proto)["MEAN"]
        print(f"## {model:7s} {proto:6s} {before[(model, proto)]:.3f} -> {r:.3f}", flush=True)
