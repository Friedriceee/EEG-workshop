"""Why is NeuralBench's EEGNet at chance? Reproduce its recipe, then change one factor at a time.
6 train / 2 val / 2 test subjects, 3 random subject splits, balanced acc per (subject, session) cell."""
import sys, numpy as np, torch, torch.nn as nn
from braindecode.models import EEGNet
from sklearn.metrics import balanced_accuracy_score as bal

D = np.load("results/proteus_epochs.npz", allow_pickle=True)
X, y = D["X"], D["y"]; sub = D["subject"].astype(str); cell = np.array([f"{a}/{b}" for a, b in zip(sub, D["session"].astype(str))])
dev = "mps"

def prep(x, mode):
    if mode == "raw": return x
    if mode == "ztrial":
        x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6); return np.clip(x, -10, 10)
    if mode == "robust":  # per-channel robust scaler fit on the whole set (like RobustScaler+clamp 20)
        med = np.median(x, axis=(0, 2), keepdims=True); q = np.percentile(x, [25, 75], axis=(0, 2), keepdims=True)
        return np.clip((x - med) / (q[1] - q[0] + 1e-6), -20, 20)

def run(tr, va, te, mode, lr, onecycle, seed, epochs=40, patience=5):
    torch.manual_seed(seed); np.random.seed(seed)
    Xp = prep(X, mode).astype(np.float32); T = lambda a: torch.tensor(a, device=dev)
    m = EEGNet(n_chans=41, n_outputs=3, n_times=480, sfreq=120).to(dev)
    w = torch.tensor(len(tr) / (3 * np.bincount(y[tr])), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    spe = int(np.ceil(len(tr) / 64))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.1) if onecycle else None
    xt, yt = T(Xp[tr]), T(y[tr]).long()
    best, bs, bad = -1, None, 0
    for ep in range(epochs):
        m.train()
        for b in torch.randperm(len(xt), device=dev).split(64):
            opt.zero_grad(); lossf(m(xt[b]), yt[b]).backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if sch: sch.step()
        m.eval()
        with torch.no_grad(): pv = m(T(Xp[va])).argmax(1).cpu().numpy()
        s = np.mean([bal(y[va][cell[va] == c], pv[cell[va] == c]) for c in set(cell[va])])
        if s > best: best, bad, bs = s, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): pt = m(T(Xp[te])).argmax(1).cpu().numpy()
    return np.mean([bal(y[te][cell[te] == c], pt[cell[te] == c]) for c in set(cell[te])]), best, ep + 1

VARIANTS = {"official: raw uV, lr1e-4": ("raw", 1e-4, True), "raw uV, lr1e-3": ("raw", 1e-3, False),
            "robust+clamp20, lr1e-4": ("robust", 1e-4, True), "z-trial, lr1e-4": ("ztrial", 1e-4, True),
            "z-trial, lr1e-3": ("ztrial", 1e-3, False)}
subs = np.array(sorted(set(sub)))
splits = []
for s in range(3):
    p = np.random.RandomState(100 + s).permutation(subs); splits.append((p[:6], p[6:8], p[8:]))
for name, (mode, lr, oc) in VARIANTS.items():
    r = [run(np.isin(sub, a), np.isin(sub, b), np.isin(sub, c), mode, lr, oc, 33 + i) for i, (a, b, c) in enumerate(splits)]
    print(f"{name:28s} test BA {np.mean([x[0] for x in r]):.3f}  (per split {[round(x[0],3) for x in r]})  val {np.mean([x[1] for x in r]):.3f}  epochs {[x[2] for x in r]}", flush=True)
