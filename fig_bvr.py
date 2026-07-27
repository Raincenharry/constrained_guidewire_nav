import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

runs = [
    (1.158, 0.990, 'tip', 'tip_d30_s0'), (1.235, 1.119, 'max', 'd160_s0'),
    (1.326, 1.164, 'max', 'd160_s2'),    (1.665, 0.978, 'max', 'd160_s3'),
    (1.678, 1.118, 'max', 'd160_s4'),    (1.738, 0.979, 'tip', 'tip_d20_s0'),
    (2.236, 0.901, 'tip', 'tip_d30_s2'), (2.470, 0.884, 'max', 'd160_s1'),
    (3.211, 0.454, 'tip', 'tip_d30_s3'), (3.329, 0.593, 'tip', 'tip_d30_s1'),
    (4.993, 0.392, 'tip', 'tip_d20_s1'),
]
B = np.array([r[0] for r in runs]); R = np.array([r[1] for r in runs])
rho, p = spearmanr(B, R)

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for sig, mk, col, lab in [('tip', 'o', '#1f77b4', 'tip force'),
                          ('max', 's', '#d62728', 'max along device')]:
    m = [r[2] == sig for r in runs]
    ax.scatter(B[m], R[m], marker=mk, c=col, s=55, label=lab,
               edgecolor='k', linewidth=0.6, zorder=3)

ax.axhline(1.0, color='grey', lw=0.8, ls=':')
ax.set_xlabel('binding severity B at activation  (EpCost / d)')
ax.set_ylabel('insertion retention R')
ax.set_title('Task damage is predicted by binding severity')
ax.text(0.97, 0.95, 'Spearman rho = %.3f\np = %.5f\nn = %d' % (rho, p, len(runs)),
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(fc='white', ec='0.7'))
ax.legend(loc='lower left', fontsize=9, title='cost signal', title_fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig('figs/fig_bvr.png', dpi=200)
print('rho %.4f  p %.6f' % (rho, p))
