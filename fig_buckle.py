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
for f in sorted(glob.glob('data/evals/*train_seed100.csv')):
    n = f.split('/')[-1]
    if 'canary' in n or 'probe' in n:
        continue
    c = cond(n)
    if c is None:
        continue
    d = pd.read_csv(f)
    if len(d) < 100:
        continue
    rows.append(dict(cond=c, ins=d.inserted_final.mean(),
                     rate=(d.force_max > 500).mean(), run=n))
t = pd.DataFrame(rows)
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
