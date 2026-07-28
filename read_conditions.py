"""
Canonical condition level reader for evaluation CSVs.

Every number in the report comes from here rather than from a one liner.
Guards, all learned the hard way:
  - n < 100 excluded, since probes are 2 to 20 episodes (27 July, fig_buckle)
  - _epNN_ excluded, checkpoint evals are a separate analysis (28 July)
  - insertion reported as stuck rate plus conditional mean, never a single
    mean, because the two failure modes are distinct (11.8)
  - force_max maxima never reported, a maximum grows with n (section 3
    correction). Buckling is a rate at fixed n.
  - optional columns guarded, the d160 era carries hinge_cost_shadow only
"""
import glob, os, re, collections
import pandas as pd

MIN_EPISODES = 100
STUCK_MM = 50
BUCKLE_N = 500


def condition(name):
    if name.startswith('r1_'):          return 'R1 unconstrained'
    if name.startswith('r4tip_w0.001'): return 'R4 tip w0.001'
    if name.startswith('r4tip_w0.01'):  return 'R4 tip w0.01'
    if name.startswith('r4_w0.0003'):   return 'R4 max era w0.0003'
    if name.startswith('sacpid_tip_ki8'):    return 'SACPID tip ki 1e-8'
    if name.startswith('sacpid_tip_anneal'): return 'SACPID tip annealed'
    if name.startswith('sacpid_tip'):        return 'SACPID tip'
    return None


def read(pool='train', root='data/evals'):
    runs, groups = [], collections.defaultdict(list)
    for f in sorted(glob.glob('%s/*%s_seed100.csv' % (root, pool))):
        n = os.path.basename(f)
        if re.search(r'_ep\d+_', n):
            print('skip checkpoint eval   %s' % n); continue
        d = pd.read_csv(f)
        if len(d) < MIN_EPISODES:
            print('skip n=%-3d probe       %s' % (len(d), n)); continue
        c = condition(n)
        if c is None:
            print('skip unknown condition %s' % n); continue

        ins = d['inserted_final']
        going = ins[ins >= STUCK_MM]
        row = dict(
            cond=c, run=n, n=len(d),
            stuck=(ins < STUCK_MM).mean(),
            ins_all=ins.mean(),
            ins_going=going.mean() if len(going) else float('nan'),
            success=(d['end_reason'] == 'target_reached').mean(),
            buckle=(d['force_max'] > BUCKLE_N).mean(),
            force_med=d['force_max'].median(),
        )
        for col in ('shadow_tip', 'shadow_max', 'hinge_cost_shadow'):
            if col in d.columns:
                row[col] = d[col].mean()
        runs.append(row)
        groups[c].append(row)
    return pd.DataFrame(runs), groups


if __name__ == '__main__':
    for pool in ('train', 'test'):
        t, groups = read(pool)
        print('\n===== %s pool =====' % pool)
        cols = ['n', 'stuck', 'ins_all', 'ins_going', 'success',
                'buckle', 'force_med']
        if 'shadow_tip' in t.columns:
            cols.append('shadow_tip')
        print(t.groupby('cond')[cols].mean().round(3).to_string())
        print('\nper run:')
        print(t[['run'] + cols].round(3).to_string(index=False))