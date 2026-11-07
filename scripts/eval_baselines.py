"""Local validation protocols + baselines on the PROTEUS training release.

Protocols (all report mean balanced accuracy over (subject, session) cells, like the official metric):
  xsubj : leave-one-subject-out (what NeuralBench's local task approximates)
  xsess : within-subject leave-one-session-out, only P04 and P10 have >1 session
  xrun  : within-subject chronological split inside a session (train runs 1-4, test runs 5-6);
          a proxy for cross-session drift that is available for all 10 subjects
Models: riem (tangent-space + logistic regression), eegnet (braindecode, MPS).
"""
import json, sys, time
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score

D = np.load("results/proteus_epochs.npz", allow_pickle=True)
X, y = D["X"], D["y"]
sub, ses, run = D["subject"].astype(str), D["session"].astype(str), D["run"].astype(int)
cell = np.array([f"{a}/{b}" for a, b in zip(sub, ses)])
SEED = 33


def norm(x):
    """Per-trial, per-channel z-score (usable window by window in streaming) + clip artefacts."""
    x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)
    return np.clip(x, -10, 10).astype(np.float32)


def fit_predict_riem(Xtr, ytr, Xte):
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    sos = butter(4, [8, 30], btype="band", fs=120, output="sos")
    f = lambda a: sosfiltfilt(sos, a, axis=-1)
    pipe = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann"),
                         LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
    pipe.fit(f(Xtr), ytr)
    return pipe.predict(f(Xte))


def fit_predict_net(build, Xtr, ytr, Xte, epochs=30):
    """Shared training recipe for every braindecode network: same normalisation, optimiser, early stopping.
    `build(n_chans, n_times)` returns the nn.Module; only the architecture differs between models."""
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr, Xte = norm(Xtr), norm(Xte)
    # hold out 15% of train for checkpoint selection (never touches the test data)
    idx = np.random.permutation(len(Xtr)); nv = int(0.15 * len(idx))
    va, tr = idx[:nv], idx[nv:]
    m = build(Xtr.shape[1], Xtr.shape[2]).to(dev)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    T = lambda a: torch.tensor(a, device=dev)
    xt, yt, xv = T(Xtr[tr]), T(ytr[tr]).long(), T(Xtr[va])
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        m.train()
        for b in torch.randperm(len(xt), device=dev).split(64):
            opt.zero_grad(); lossf(m(xt[b]), yt[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(xv).argmax(1).cpu().numpy()
        s = balanced_accuracy_score(ytr[va], pv)
        if s > best:
            best, bad = s, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return m(T(Xte)).argmax(1).cpu().numpy()


def _braindecode(name):
    def build(n_chans, n_times):
        import braindecode.models as BM
        return getattr(BM, name)(n_chans=n_chans, n_outputs=3, n_times=n_times, sfreq=120)
    return lambda Xtr, ytr, Xte: fit_predict_net(build, Xtr, ytr, Xte)


fit_predict_eegnet = _braindecode("EEGNet")
MODELS = {"riem": fit_predict_riem, "eegnet": fit_predict_eegnet,
          "fbcnet": _braindecode("FBCNet"), "atcnet": _braindecode("ATCNet")}


def splits(protocol):
    if protocol == "xsubj":
        for s in sorted(set(sub)):
            yield s, sub != s, sub == s
    elif protocol == "xsess":
        for s in ["sub-P04", "sub-P10"]:
            for t in sorted(set(ses[sub == s])):
                yield f"{s}/{t}", (sub == s) & (ses != t), (sub == s) & (ses == t)
    elif protocol == "xrun":
        for s in sorted(set(sub)):
            yield s, (sub == s) & (run <= 4), (sub == s) & (run >= 5)


def evaluate(model, protocol):
    out = {}
    for name, tr, te in splits(protocol):
        pred = MODELS[model](X[tr], y[tr], X[te])
        # balanced accuracy per (subject, session) cell, then average cells
        cells = [balanced_accuracy_score(y[te][cell[te] == c], pred[cell[te] == c]) for c in sorted(set(cell[te]))]
        out[name] = float(np.mean(cells))
        print(f"  {model:7s} {protocol:6s} {name:14s} BA={out[name]:.3f}", flush=True)
    out["MEAN"] = float(np.mean([v for k, v in out.items()]))
    return out


if __name__ == "__main__":
    protocols = sys.argv[1].split(",")
    models = sys.argv[2].split(",")
    res = {}
    for p in protocols:
        for m in models:
            t0 = time.time()
            res[f"{m}/{p}"] = evaluate(m, p)
            print(f"== {m}/{p}: mean BA {res[f'{m}/{p}']['MEAN']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(f"results/baselines_{sys.argv[3] if len(sys.argv) > 3 else 'run'}.json", "w"), indent=1)
