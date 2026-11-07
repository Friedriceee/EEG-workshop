"""Distribution profile of the PROTEUS training epochs (what the data looks like, and how big the shifts are)."""
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats import f_oneway
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.distance import distance_riemann

D = np.load("results/proteus_epochs.npz", allow_pickle=True)
X, y = D["X"], D["y"]
sub, ses, run, task = D["subject"].astype(str), D["session"].astype(str), D["run"].astype(int), D["task"].astype(str)
ch = [str(c) for c in D["ch_names"]]
cell = np.array([f"{a}/{b[-1]}" for a, b in zip(sub, ses)])
cells = sorted(set(cell))
FS = 120
NAMES = ["MI", "SUB", "WORD"]

print("== 1. Amplitude (per-trial per-channel std, uV) and artefacts")
sd = X.std(-1)                       # (trials, channels)
tr_sd = np.median(sd, axis=1)        # robust per-trial level
tr_max = np.abs(X).max(axis=(1, 2))
rows = []
for c in cells:
    m = cell == c
    rows.append((c, m.sum(), np.median(tr_sd[m]), np.percentile(tr_sd[m], 90), (tr_max[m] > 200).mean()))
t = pd.DataFrame(rows, columns=["cell", "n", "median_sd_uV", "p90_sd_uV", "frac_|x|>200uV"])
print(t.round(3).to_string(index=False))
print(f"overall: median sd {np.median(tr_sd):.1f} uV, trials with |x|>200 uV: {(tr_max>200).mean():.1%}, |x|>1000 uV: {(tr_max>1000).mean():.1%}")
per_sub = pd.Series(tr_sd).groupby(sub).median()
print(f"between-subject spread of median sd: min {per_sub.min():.1f}  max {per_sub.max():.1f}  (ratio {per_sub.max()/per_sub.min():.1f}x)")

print("\n== 2. Band power by class (relative power, trials z-scored per subject-session, ANOVA over 3 classes)")
f, P = welch(X, fs=FS, nperseg=FS, axis=-1)          # (trials, ch, freq)
bands = {"delta 1-4": (1, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13), "beta 13-30": (13, 30), "gamma 30-45": (30, 45)}
tot = P[..., (f >= 1) & (f < 45)].sum(-1)
for name, (lo, hi) in bands.items():
    bp = np.log(P[..., (f >= lo) & (f < hi)].sum(-1) / tot + 1e-12)   # (trials, ch)
    z = bp.copy()
    for c in cells:                    # remove session level so we see class effect, not session effect
        m = cell == c
        z[m] = (bp[m] - bp[m].mean(0)) / (bp[m].std(0) + 1e-9)
    Fs = np.array([f_oneway(*[z[y == k][:, i] for k in range(3)])[0] for i in range(len(ch))])
    top = np.argsort(-Fs)[:3]
    print(f"{name:12s} mean F over channels {Fs.mean():.2f}; strongest: " + ", ".join(f"{ch[i]} (F={Fs[i]:.1f})" for i in top))

print("\n== 3. How big is each kind of shift? (Riemannian distance between mean covariances, per-trial-normalised)")
Xn = X / (X.std(-1, keepdims=True) + 1e-6)
Xn = np.clip(Xn, -10, 10)
C = Covariances(estimator="oas").transform(Xn)
M = {c: mean_covariance(C[cell == c], metric="riemann") for c in cells}
Mr = {}
for c in cells:                       # per-run means to measure within-session drift
    for r in sorted(set(run[cell == c])):
        m = (cell == c) & (run == r)
        if m.sum() > 5:
            Mr[(c, r)] = mean_covariance(C[m], metric="riemann")
within_run, same_sub_diff_ses, diff_sub = [], [], []
keys = list(Mr)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        (ci, ri), (cj, rj) = keys[i], keys[j]
        d = distance_riemann(Mr[keys[i]], Mr[keys[j]])
        if ci == cj: within_run.append(d)
        elif ci.split("/")[0] == cj.split("/")[0]: same_sub_diff_ses.append(d)
        else: diff_sub.append(d)
for nm, v in [("same session, different run", within_run), ("same subject, different session", same_sub_diff_ses), ("different subject", diff_sub)]:
    print(f"{nm:34s} mean distance {np.mean(v):.2f}  (n pairs {len(v)})")
D_s = np.array([[distance_riemann(M[a], M[b]) for b in cells] for a in cells])

print("\n== 4. Class balance and run types")
print(pd.crosstab(task, y).rename(columns=dict(enumerate(NAMES))).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
cols = ["#1f77b4", "#d95f02", "#1b9e77"]
for k in range(3):
    for chn, ls in [("C3", "-"), ("C4", "--")]:
        ax[0].plot(f, np.log10(P[y == k][:, ch.index(chn)].mean(0)), ls, color=cols[k], label=f"{NAMES[k]} {chn}")
ax[0].set_xlim(1, 45); ax[0].set_xlabel("Hz"); ax[0].set_ylabel("log10 power (uV^2/Hz)"); ax[0].set_title("Spectrum by class (C3 solid, C4 dashed)"); ax[0].legend(fontsize=7)
order = np.argsort(np.array([int(c.split('/')[0][-2:]) * 10 + int(c.split('/')[1]) for c in cells]))
ax[1].boxplot([np.log10(tr_sd[cell == cells[i]]) for i in order], tick_labels=[cells[i].replace("sub-", "") for i in order], showfliers=False)
ax[1].set_title("Signal level per subject/session (log10 sd, uV)"); ax[1].tick_params(axis="x", rotation=90, labelsize=7)
im = ax[2].imshow(D_s[np.ix_(order, order)], cmap="viridis")
ax[2].set_xticks(range(len(cells))); ax[2].set_yticks(range(len(cells)))
ax[2].set_xticklabels([cells[i].replace("sub-", "") for i in order], rotation=90, fontsize=6); ax[2].set_yticklabels([cells[i].replace("sub-", "") for i in order], fontsize=6)
ax[2].set_title("Distance between session mean covariances"); plt.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.savefig("results/data_profile.png", dpi=130)
print("saved results/data_profile.png")
