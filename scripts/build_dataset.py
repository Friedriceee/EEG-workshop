"""Epoch the PROTEUS training release into one npz (official-style preprocessing).

Official task config: resample 120 Hz, band 0.1-75 Hz, notch 50/60 Hz, windows 1.0-5.0 s after cue,
scale V->uV. Output X: (n_trials, 41, 480) float32 in uV, plus subject/session/run/label metadata.
"""
import glob, os, sys
import numpy as np, pandas as pd, mne

mne.set_log_level("ERROR")
ROOT = sys.argv[1]
OUT = sys.argv[2]
LABELS = {"mi": 0, "sub": 1, "word": 2}
SFREQ, TMIN, TMAX = 120, 1.0, 5.0

X, meta, ch_ref = [], [], None
for edf in sorted(glob.glob(f"{ROOT}/sub-*/ses-*/eeg/*_eeg.edf")):
    ev_path = edf.replace("_eeg.edf", "_events.tsv")
    ev = pd.read_csv(ev_path, sep="\t")
    ev = ev[ev.trial_type.isin(LABELS)]
    if ev.empty:
        continue
    parts = os.path.basename(edf).split("_")
    sub, ses = parts[0], parts[1]
    task, run = parts[2].replace("task-", ""), parts[3].replace("run-", "")
    raw = mne.io.read_raw_edf(edf, preload=True)
    raw.pick("eeg")
    if ch_ref is None:
        ch_ref = raw.ch_names
    assert raw.ch_names == ch_ref, f"channel mismatch in {edf}"
    raw.filter(0.1, 75.0, verbose=False)
    raw.notch_filter([50.0, 60.0], verbose=False)
    raw.resample(SFREQ)
    data = raw.get_data() * 1e6
    n = int((TMAX - TMIN) * SFREQ)
    for _, r in ev.iterrows():
        s = int(round((r.onset + TMIN) * SFREQ))
        if s + n > data.shape[1]:
            continue
        X.append(data[:, s:s + n].astype(np.float32))
        meta.append((sub, ses, task, int(run), r.onset, LABELS[r.trial_type]))
    print(sub, ses, task, run, len(ev), flush=True)

X = np.stack(X)
m = pd.DataFrame(meta, columns=["subject", "session", "task", "run", "onset", "y"])
np.savez_compressed(OUT, X=X, y=m.y.values, subject=m.subject.values, session=m.session.values,
                    task=m.task.values, run=m.run.values, onset=m.onset.values, ch_names=np.array(ch_ref))
print("X", X.shape, "ch", len(ch_ref), "std(uV)", float(X.std()))
