"""Summarise an episode CSV written by steve_cmdp.

Usage:  python analyse.py runs_x/episodes_seed0.csv [window] [skip]
        window  episodes in the reporting tail, default 200
        skip    rows dropped from the front of that tail, default 0
"""
import sys
import pandas as pd

path = sys.argv[1]
window = int(sys.argv[2]) if len(sys.argv) > 2 else 200
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 0

df = pd.read_csv(path)
print("rows in file: %d" % len(df))

# Filter to the training instance BEFORE any tail. OmniSafe writes two
# environments into one CSV and both count episodes from zero.
if "env_instance" in df:
    main_inst = df.env_instance.value_counts().idxmax()
    train = df[df.env_instance == main_inst].sort_values("episode").reset_index(drop=True)
    print("env instances: %s, using %s as training"
          % (df.env_instance.value_counts().to_dict(), main_inst))
else:
    train = df.sort_values("episode").reset_index(drop=True)
print("training episodes: %d" % len(train))

last = train.tail(window)
if skip:
    last = last.iloc[skip:]
print()
print("reporting window: tail(%d) skip %d -> %d episodes" % (window, skip, len(last)))
print("  success rate        %.3f" % last.success.mean())
succ = last[last.success == 1]
if len(succ):
    print("  steps when success  %.1f" % succ.steps.mean())
print("  inserted_final      %.1f mm" % last.inserted_final.mean())
print("  mean force          %.3f N" % last.force_mean.mean())
print("  p95 force           %.3f N" % last.force_p95.mean())
print("  force_max median    %.3f N" % last.force_max.median())
print("  force_max p90       %.3f N" % last.force_max.quantile(0.90))
print("  buckling rate >500N %.3f   (rate, comparable across n)"
      % (last.force_max > 500).mean())
print("  steps over 0.85 N   %.1f per episode" % last.steps_over_threshold.mean())
print("  clamp steps         %.1f per episode" % last.clamp_steps.mean())
print("  end reasons         %s" % last.end_reason.value_counts().to_dict())

# Cost columns, guarded. The d160 era CSVs carry only hinge_cost_shadow.
print()
print("cost columns present in this file")
for col in ("ep_cost", "shadow_tip", "shadow_max", "hinge_cost_shadow", "ep_penalty"):
    if col not in last.columns:
        continue
    pooled = last[col].sum() / last.steps.sum()
    med = (last[col] / last.steps).median()
    print("  %-18s per episode %8.2f   per step pooled %.4f   per step median %.4f"
          % (col, last[col].mean(), pooled, med))
if "cost_fn" in last.columns:
    print("  cost_fn %s   reward_penalty_weight %s"
          % (last.cost_fn.iloc[-1],
             last.reward_penalty_weight.iloc[-1] if "reward_penalty_weight" in last else "n/a"))

# Block table over the whole training run.
block_n = 200
print()
print("blocks of %d episodes" % block_n)
cost_col = next((c for c in ("shadow_tip", "hinge_cost_shadow", "ep_cost")
                 if c in train.columns), None)
for i in range(0, len(train), block_n):
    b = train.iloc[i:i + block_n]
    if len(b) < block_n // 2:
        continue
    cs = ""
    if cost_col:
        cs = "   %s/step %.4f" % (cost_col, b[cost_col].sum() / b.steps.sum())
    print("  ep %5d to %5d  success %.3f  ins %6.1f  buckle %.3f  ret %+.3f%s"
          % (b.episode.iloc[0], b.episode.iloc[-1], b.success.mean(),
             b.inserted_final.mean(), (b.force_max > 500).mean(),
             b.ep_return.mean(), cs))

if "arch_type" in last:
    print()
    print("success by arch type in the window")
    for arch, grp in last.groupby("arch_type"):
        print("  %-14s n=%3d  success %.3f  ins %6.1f"
              % (arch, len(grp), grp.success.mean(), grp.inserted_final.mean()))
