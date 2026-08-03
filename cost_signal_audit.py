import glob, os
import numpy as np
import pandas as pd

DROP = ('r1canary', 'run_epoch', '_ep30')

rows, skipped = [], []
for f in sorted(glob.glob('evals/*_test_seed100.csv')):
    run = os.path.basename(f).replace('_test_seed100.csv', '')
    if any(k in run for k in DROP):
        skipped.append((run, 'excluded name'))
        continue
    cols = pd.read_csv(f, nrows=0).columns
    if 'shadow_max' not in cols:
        skipped.append((run, 'no shadow_max'))
        continue
    df = pd.read_csv(f)
    df['run'] = run
    rows.append(df)
d = pd.concat(rows, ignore_index=True)

for run, why in skipped:
    print('SKIP %-28s %s' % (run, why))
print()
print('episodes', len(d), 'runs', d['run'].nunique())

cols = ['force_max', 'shadow_max', 'shadow_tip', 'tip_mean', 'tip_max',
        'excess_max', 'inserted_final']
print()
print(d[cols].describe().T.to_string())

print()
print('correlation with inserted_final')
for c in ['force_max', 'shadow_max', 'tip_max', 'tip_mean']:
    ok = d[[c, 'inserted_final']].dropna()
    if len(ok) > 2:
        print('  %-12s pearson %+.3f   spearman %+.3f   n %d' % (
            c,
            ok[c].corr(ok['inserted_final']),
            ok[c].corr(ok['inserted_final'], method='spearman'),
            len(ok)))

print()
print('binned by insertion depth, mean of each signal')
bins = [0, 100, 200, 300, 400, 500, 10000]
d['bin'] = pd.cut(d['inserted_final'], bins)
print(d.groupby('bin', observed=True)[
    ['force_max', 'shadow_max', 'tip_max', 'tip_mean']].agg(['mean', 'count']).round(3).to_string())
