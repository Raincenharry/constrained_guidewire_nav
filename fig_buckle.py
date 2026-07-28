import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def cond(name):
    if name.startswith('r1_'):            return 'R1 unconstrained'
    if name.startswith('r4tip_w0.001'):   return 'R4 tip w0.001'
    if name.startswith('r4_w0.0003'):     return 'R4 max era w0.0003'
    if name.startswith('sacpid_tip'):     return 'SACPID tip'
    return None

rows = []
import re
for f in sorted(glob.glob('data/evals/*train_seed100.csv')):
    n = f.split('/')[-1]
    # Checkpoint evaluations (sacpid_..._ep30_...) are a separate seed
    # dependent analysis and belong in their own figure, not this one.
    if re.search(r'_ep\d+_', n):
        print('excluded, checkpoint eval: %s' % n)
        continue
    c = cond(n)
    if c is None:
        print('excluded, unrecognised condition: %s' % n)
        continue
    d = pd.read_csv(f)
    # n=100 is the protocol. Probe and canary runs are 2 to 20 episodes and
    # would weight equally with a full run. This is the guard that matters;
    # the name based skip above is a convenience, not the safeguard.
    if len(d) < 100:
        print('excluded, n=%d not a full evaluation: %s' % (len(d), n))
        continue
    rows.append(dict(cond=c, ins=d.inserted_final.mean(),
                     rate=(d.force_max > 500).mean(), run=n))
t = pd.DataFrame(rows)

expected = {'R1 unconstrained': 5, 'R4 tip w0.001': 2,
            'R4 max era w0.0003': 3, 'SACPID tip': 6}
got = t.groupby('cond').size().to_dict()
for c, k in expected.items():
    if got.get(c, 0) != k:
        print('WARNING %s: expected %d runs, got %d' % (c, k, got.get(c, 0)))
print(t.groupby('cond')[['ins', 'rate']].mean().round(3))

sp = t[t['cond'] == 'SACPID tip']
rho, p = spearmanr(sp.ins, sp.rate)
print('SACPID insertion vs buckling rate: rho %.3f  p %.4f' % (rho, p))

style = {'R1 unconstrained':   ('#d62728', 'o'),
         'R4 tip w0.001':      ('#2ca02c', 's'),
         'R4 max era w0.0003': ('#8c564b', 'D'),
         'SACPID tip':         ('#1f77b4', '^')}
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for c, (col, mk) in style.items():
    g = t[t['cond'] == c]
    if len(g):
        ax.scatter(g.ins, g.rate, c=col, marker=mk, s=55, label=c,
                   edgecolor='k', linewidth=0.6, zorder=3)
ax.axvspan(180, 215, color='grey', alpha=0.12, zorder=0)
ax.text(197, ax.get_ylim()[1]*0.93, 'matched insertion', ha='center',
        fontsize=8, color='0.35')
ax.set_xlabel('mean insertion (mm)')
ax.set_ylabel('fraction of episodes with force_max > 500 N')
ax.set_title('Buckling rate, train pool, n = 100 per run')
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('figs/fig_buckle.png', dpi=200)
