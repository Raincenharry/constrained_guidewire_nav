import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DROP = ('r1canary', 'run_epoch', '_ep30')
CEIL = 5.0

rows = []
for f in sorted(glob.glob('evals/*_test_seed100.csv')):
    run = os.path.basename(f).replace('_test_seed100.csv', '')
    if any(k in run for k in DROP):
        continue
    if 'shadow_max' not in pd.read_csv(f, nrows=0).columns:
        continue
    df = pd.read_csv(f)
    df['run'] = run
    rows.append(df)
d = pd.concat(rows, ignore_index=True)
print('episodes', len(d), 'runs', d['run'].nunique())

EDGES = np.arange(50, 401, 50)
d['bin'] = pd.cut(d['inserted_final'], EDGES)
ctr = (EDGES[:-1] + EDGES[1:]) / 2.0

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

for col, lab, c in [('force_max', 'max along device', 'tab:purple'),
                    ('tip_max',   'tip force'      ,    'tab:green')]:
    g = d.groupby('bin', observed=True)[col]
    med = g.median().values
    lo, hi = g.quantile(0.25).values, g.quantile(0.75).values
    x = ctr[:len(med)]
    ax[0].plot(x, med, 'o-', color=c, lw=2, ms=4, label=lab)
    ax[0].fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)

ax[0].set_yscale('log')
ax[0].axhline(CEIL, color='0.3', ls='--', lw=1.2)
ax[0].text(0.03, 0.30, 'hinge ceiling 5 N\n(threshold 0.85 N)',
           transform=ax[0].transAxes, ha='left', fontsize=8, color='0.3')
ax[0].set_ylabel('peak force (N)')
ax[0].set_xlabel('insertion depth at episode end (mm)')
ax[0].set_title('a. peak force per episode (N)')
ax[0].legend(fontsize=8)

for col, lab, c in [('shadow_max', 'cost from max along device', 'tab:purple'),
                    ('shadow_tip', 'cost from tip force'    ,    'tab:green')]:
    g = d.groupby('bin', observed=True)[col]
    med = g.median().values
    lo, hi = g.quantile(0.25).values, g.quantile(0.75).values
    x = ctr[:len(med)]
    ax[1].plot(x, med, 'o-', color=c, lw=2, ms=4, label=lab)
    ax[1].fill_between(x, np.maximum(lo, 0.1), hi, color=c, alpha=0.15, lw=0)

for a in ax:
    a.set_xlim(50, 400)

ax[1].set_yscale('log')
ax[1].set_ylim(0.1, 1500)
ax[1].set_ylabel('accumulated cost (newton steps)')
ax[1].set_xlabel('insertion depth at episode end (mm)')
ax[1].set_title('b. accumulated cost per episode (newton steps)')
ax[1].legend(fontsize=8)

fig.tight_layout()
fig.savefig('figs_19_aug/figE_cost_signal.png', dpi=300)
print('wrote figs_19_aug/figE_cost_signal.png')

print()
print(d.groupby('bin', observed=True)[
    ['force_max', 'tip_max', 'shadow_max', 'shadow_tip']].median().round(3).to_string())
