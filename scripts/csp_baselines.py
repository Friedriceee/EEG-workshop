"""CSP-based classical baselines on the same three exams as eval_baselines.py.

csp_lda : multi-class CSP (6 components, Ledoit-Wolf covariance) on 8-30 Hz  -> shrinkage LDA
csp_rda : same CSP features                                                   -> regularised QDA (reg_param=0.5), i.e. RDA-style
Usage: python scripts/csp_baselines.py xrun,xsess,xsubj csp_lda,csp_rda csp
"""
import json, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP

sys.path.insert(0, "scripts")
import eval_baselines as E

SOS = butter(4, [8, 30], btype="band", fs=120, output="sos")


def make(clf):
    def fit_predict(Xtr, ytr, Xte):
        f = lambda a: sosfiltfilt(SOS, a, axis=-1).astype(np.float64)
        pipe = make_pipeline(CSP(n_components=6, reg="ledoit_wolf", log=True), clf())
        pipe.fit(f(Xtr), ytr)
        return pipe.predict(f(Xte))
    return fit_predict


E.MODELS["csp_lda"] = make(lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
E.MODELS["csp_rda"] = make(lambda: QuadraticDiscriminantAnalysis(reg_param=0.5))

if __name__ == "__main__":
    res = {}
    for p in sys.argv[1].split(","):
        for m in sys.argv[2].split(","):
            res[f"{m}/{p}"] = E.evaluate(m, p)
            print(f"== {m}/{p}: mean BA {res[f'{m}/{p}']['MEAN']:.3f}", flush=True)
    json.dump(res, open(f"results/baselines_{sys.argv[3]}.json", "w"), indent=1)
