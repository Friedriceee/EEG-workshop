"""Streaming Riemannian solver: per-session causal Euclidean alignment + tangent space + logistic regression.

Inference is pure numpy (weights come from weights.pt, a dict of tensors/lists written by scripts/train_submission.py).
Per window (channels x samples, uV, 120 Hz):
  1. keep the 41 training channels (by name when meta gives names)
  2. update the running spatial covariance of THIS stream (4-40 Hz) with the current window, whiten the window with it
  3. band-pass 8-30 Hz, OAS covariance, tangent space at the subject's reference, linear classifier
reset_state() must be called at the start of every (subject, session) stream.

UNVERIFIED against the official kit: the real `CompetSolver` base class and the `meta` keys are not available locally
(they live in the Benchopt competition kit). `Solver` at the bottom is a placeholder adapter, NOT the official interface.
Known limits: only sfreq == 120; input assumed in uV; subject id is read from meta["subject"] (assumed key).
"""
import numpy as np

CLASSES = ("mi", "sub", "word")


def _odd_ext(x, n):
    left = 2 * x[..., :1] - x[..., n:0:-1]
    right = 2 * x[..., -1:] - x[..., -2:-n - 2:-1]
    return np.concatenate([left, x, right], axis=-1)


def _sosfilt(sos, x, zi_scaled):
    """Direct-form-II-transposed biquad cascade along the last axis, per-section initial states zi_scaled (S, 2, C)."""
    y = x.copy()
    for k, (b0, b1, b2, a0, a1, a2) in enumerate(sos):
        z0, z1 = zi_scaled[k, 0].copy(), zi_scaled[k, 1].copy()
        out = np.empty_like(y)
        for t in range(y.shape[-1]):
            xt = y[:, t]
            yt = b0 * xt + z0
            z0 = b1 * xt - a1 * yt + z1
            z1 = b2 * xt - a2 * yt
            out[:, t] = yt
        y = out
    return y


def sosfiltfilt(sos, zi, x):
    """numpy re-implementation of scipy.signal.sosfiltfilt (default odd padding), x: (C, T)."""
    n_sec = sos.shape[0]
    padlen = 3 * (2 * n_sec + 1 - min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum())))
    ext = _odd_ext(x, padlen)
    y = _sosfilt(sos, ext, zi[:, :, None] * ext[None, None, :, 0])
    y = _sosfilt(sos, y[:, ::-1], zi[:, :, None] * y[None, None, :, -1])[:, ::-1]
    return y[:, padlen:-padlen]


def _eig_fun(M, f):
    w, v = np.linalg.eigh(M)
    return (v * f(w)) @ v.T


def _oas_cov(x):
    """sklearn OAS shrinkage covariance of x (C, T) (what pyriemann Covariances(estimator='oas') computes)."""
    c, t = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    emp = xc @ xc.T / t
    mu = np.trace(emp) / c
    alpha = np.mean(emp ** 2)
    num = alpha + mu ** 2
    den = (t + 1.0) * (alpha - mu ** 2 / c)
    shrink = 1.0 if den == 0 else min(num / den, 1.0)
    return (1.0 - shrink) * emp + shrink * mu * np.eye(c)


class RiemannianStreamSolver:
    def __init__(self, weights):
        self.w = weights
        self.sos_a, self.zi_a = np.asarray(weights["sos_align"]), np.asarray(weights["zi_align"])
        self.sos_b, self.zi_b = np.asarray(weights["sos_band"]), np.asarray(weights["zi_band"])
        self.ch = list(weights["ch_names"])
        self.iu = np.triu_indices(len(self.ch))
        self.coefs = np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0))
        self.reset_state()

    @classmethod
    def load(cls, path):
        import torch
        raw = torch.load(path, weights_only=True)
        return cls({k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in raw.items()})

    def reset_state(self):
        self._cov_sum = np.zeros((len(self.ch), len(self.ch)))
        self._n = 0

    def _model_for(self, subject):
        key = subject if f"{subject}/coef" in self.w else "generic"
        return self.w[f"{key}/ref"], self.w[f"{key}/coef"], self.w[f"{key}/intercept"]

    def predict_proba(self, x, meta=None):
        meta = meta or {}
        if meta.get("sfreq", 120) != 120:
            raise ValueError("only 120 Hz windows are supported")
        names = meta.get("ch_names")
        if names is not None:
            x = x[[list(names).index(c) for c in self.ch]]
        elif x.shape[0] != len(self.ch):
            raise ValueError(f"got {x.shape[0]} channels and no ch_names; expected {len(self.ch)}")
        x = np.asarray(x, dtype=np.float64)
        xb = sosfiltfilt(self.sos_a, self.zi_a, x)
        self._cov_sum += xb @ xb.T / xb.shape[1]
        self._n += 1
        R = self._cov_sum / self._n
        xa = _eig_fun(R, lambda w: 1.0 / np.sqrt(w + 1e-6 * w.max())) @ x
        C = _oas_cov(sosfiltfilt(self.sos_b, self.zi_b, xa))
        ref, coef, intercept = self._model_for(meta.get("subject"))
        cm12 = _eig_fun(ref, lambda w: 1.0 / np.sqrt(w))
        s = _eig_fun(cm12 @ C @ cm12, np.log)
        logits = coef @ (self.coefs * s[self.iu]) + intercept
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict(self, x, meta=None):
        return int(np.argmax(self.predict_proba(x, meta)))


class Solver:  # PLACEHOLDER: replace with `class Solver(CompetSolver)` once the official kit is available
    def __init__(self, weights_path="weights.pt"):
        self.model = RiemannianStreamSolver.load(weights_path)

    def reset_state(self):
        self.model.reset_state()

    def predict(self, window, meta=None):
        return self.model.predict(window, meta)
