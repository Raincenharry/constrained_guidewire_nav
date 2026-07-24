"""Does tip angle lead force? Cross correlation at lags 1 to 20 steps."""
import numpy as np, pandas as pd

THRESH = 0.85
d = pd.read_csv("angle_trace.csv")
print("steps: %d over %d episodes" % (len(d), d.episode.nunique()))
print("steps over %.2f N: %d (%.1f %%)"
      % (THRESH, (d.force_max > THRESH).sum(), 100 * (d.force_max > THRESH).mean()))
print("force_max  median %.3f  p95 %.3f  max %.1f"
      % (d.force_max.median(), d.force_max.quantile(0.95), d.force_max.max()))
print("angle_bend median %.2f  p95 %.2f  max %.1f"
      % (d.angle_bend.median(), d.angle_bend.quantile(0.95), d.angle_bend.max()))
print()

angles = ["angle_bend", "angle_bend_span"]

print("Pearson r between angle at t and force at t+lag, pooled within episodes")
print("positive lag means angle leads force")
for a in angles:
    out = []
    for lag in range(-5, 21):
        xs, ys = [], []
        for _, g in d.groupby("episode"):
            x, y = g[a].values, g.force_max.values
            if lag >= 0:
                if len(x) <= lag: continue
                xs.append(x[:len(x) - lag] if lag else x); ys.append(y[lag:])
            else:
                k = -lag
                if len(x) <= k: continue
                xs.append(x[k:]); ys.append(y[:len(y) - k])
        x, y = np.concatenate(xs), np.concatenate(ys)
        out.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    print()
    print(a)
    for lag, v in zip(range(-5, 21), out):
        print("   lag %+3d   r %+.3f" % (lag, v))
    print("   peak |r| at lag %+d" % (int(np.nanargmax(np.abs(out))) - 5))

print()
print("Mean angle in the N steps before a spike, against all other steps")
for a in angles:
    for lead in (1, 3, 5, 10):
        pre, other = [], []
        for _, g in d.groupby("episode"):
            spike = (g.force_max.values > THRESH)
            ang = g[a].values
            mask = np.zeros(len(ang), bool)
            for i in np.where(spike)[0]:
                mask[max(0, i - lead):i] = True
            mask &= ~spike
            pre.append(ang[mask]); other.append(ang[~mask & ~spike])
        pre, other = np.concatenate(pre), np.concatenate(other)
        if len(pre) < 20: continue
        print("%-18s lead %2d   before spike %6.2f deg   elsewhere %6.2f deg   ratio %.2f"
              % (a, lead, pre.mean(), other.mean(), pre.mean() / max(other.mean(), 1e-9)))
