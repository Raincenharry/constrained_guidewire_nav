import glob, os
import pandas as pd

rows = []
for d in sorted(glob.glob('traces/runs_sacpid_tip_*')):
    hits = glob.glob(os.path.join(d, '**', 'progress.csv'), recursive=True)
    if not hits:
        continue
    df = pd.read_csv(hits[0])
    lam = df['Metrics/LagrangeMultiplier']
    cost = df['Metrics/EpCost']
    active = lam.gt(0)
    first = int(active.idxmax()) if active.any() else -1
    lpost = lam[first:] if first >= 0 else lam
    cpost = cost[first:] if first >= 0 else cost
    rows.append(dict(
        run=os.path.basename(d).replace('runs_sacpid_tip_', ''),
        n=len(df),
        act=first,
        lam_max=round(lpost.max(), 5),
        lam_mean=round(lpost.mean(), 5),
        lam_std=round(lpost.std(), 5),
        lam_end=round(lam.iloc[-1], 5),
        cost_post=round(cpost.mean(), 2),
        cost_last10=round(cost.iloc[-10:].mean(), 2),
    ))

print(pd.DataFrame(rows).to_string(index=False))
