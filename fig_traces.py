import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDS = [
    ('w20_d20',  20,  'tab:red',   'd = 20'),
    ('w20_d30',  30,  'tab:orange','d = 30'),
    ('w20_d150', 150, 'tab:blue',  'd = 150'),
]
NEPOCH = 50

# only seeds present in the eval evidence base, see section 1 seed counts
KEEP = {'w20_d150': ['s2', 's3', 's4']}

def load(tag):
    out = []
    for d in sorted(glob.glob('traces/runs_sacpid_tip_' + tag + '_s*')):
        hits = glob.glob(os.path.join(d, '**', 'progress.csv'), recursive=True)
        if not hits:
            continue
        keep = KEEP.get(tag)
        if keep is not None and os.path.basename(d).split('_')[-1] not in keep:
            print('SKIP not in evidence base', os.path.basename(d))
            continue
        df = pd.read_csv(hits[0])
        if len(df) < NEPOCH:
            print('SKIP short run', os.path.basename(d), len(df))
            continue
        out.append(df)
    return out

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

for tag, budget, col, lab in CONDS:
    runs = load(tag)
    if not runs:
        continue
    cost = np.vstack([r['Metrics/EpCost'].values[:NEPOCH] for r in runs])
    lam  = np.vstack([r['Metrics/LagrangeMultiplier'].values[:NEPOCH] for r in runs])
    ep = np.arange(NEPOCH)
    n = cost.shape[0]

    clo, chi = np.percentile(cost, [25, 75], axis=0)
    ax[0].plot(ep, np.median(cost, axis=0), color=col, lw=2, label=lab + ' (%d)' % n)
    ax[0].fill_between(ep, clo, chi, color=col, alpha=0.15, lw=0)
    ax[0].axhline(budget, color=col, ls=':', lw=1.2)

    llo, lhi = np.percentile(lam, [25, 75], axis=0)
    ax[1].plot(ep, np.median(lam, axis=0), color=col, lw=2, label=lab + ' (%d)' % n)
    ax[1].fill_between(ep, np.maximum(llo, 3e-7), lhi, color=col, alpha=0.15, lw=0)

for a in ax:
    a.axvspan(0, 20, color='0.85', alpha=0.6, lw=0)
    a.set_xlabel('training epoch')
    a.set_xlim(0, NEPOCH - 1)
    a.legend(fontsize=8)

ax[0].set_yscale('log')
ax[0].set_ylabel('training rollout cost (newton steps)')
ax[0].set_title('a. cost against budget, dotted line is d')
ax[1].set_yscale('symlog', linthresh=1e-6)
ax[1].set_ylim(0, 0.2)
ax[1].set_ylabel('Lagrange multiplier')
ax[1].set_title('b. multiplier, shaded region is warmup')

fig.tight_layout()
os.makedirs('figs_31_july', exist_ok=True)
fig.savefig('figs_31_july/figD_traces.png', dpi=200)
print('wrote figs_31_july/figD_traces.png')

for tag, budget, col, lab in CONDS:
    runs = load(tag)
    cost = np.vstack([r['Metrics/EpCost'].values[:NEPOCH] for r in runs])
    print('%-10s n=%d  last10 mean %.2f  budget %d' %
          (tag, cost.shape[0], cost[:, -10:].mean(), budget))
