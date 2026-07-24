"""Does angle lead force? Trend removed, wider lag range, step differences."""
import numpy as np, pandas as pd

THRESH = 0.85
d = pd.read_csv("angle_trace.csv")
angles = [c for c in ("angle_bend_span", "angle_centerline") if c in d.columns]

def xcorr(d, a, col, lags):
    out = []
    for lag in lags:
        xs, ys = [], []
        for _, g in d.groupby("episode"):
            x, y = g[a].values, g[col].values
            x = x - x.mean(); y = y - y.mean()          # per episode detrend
            if lag >= 0:
                if len(x) <= lag: continue
                xs.append(x[:len(x) - lag] if lag else x); ys.append(y[lag:])
            else:
                k = -lag
                if len(x) <= k: continue
                xs.append(x[k:]); ys.append(y[:len(y) - k])
        x, y = np.concatenate(xs), np.concatenate(ys)
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    return np.array(out)

d["force_clip"] = d.force_max.clip(upper=5.0)           # same ceiling as the cost
d["dforce"] = d.groupby("episode").force_clip.diff().fillna(0)
for a in angles:
    d["d_" + a] = d.groupby("episode")[a].diff().fillna(0)

lags = np.arange(-20, 21)
for a in angles:
    for col, label in [("force_clip", "level"), ("dforce", "change")]:
        src = a if col == "force_clip" else "d_" + a
        r = xcorr(d, src, col, lags)
        peak = lags[int(np.nanargmax(np.abs(r)))]
        print("%-18s vs force %-7s  peak lag %+3d  r %+.3f   r at +1 %+.3f  +5 %+.3f"
              % (a, label, peak, r[int(np.nanargmax(np.abs(r)))],
                 r[list(lags).index(1)], r[list(lags).index(5)]))
print()
print("positive lag means angle leads force. negative means force leads angle.")
