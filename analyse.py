"""Summarise an episode CSV written by steve_cmdp.

Usage:  python analyse.py runs_r1_baseline/episodes_seed0.csv [window]
"""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
window = int(sys.argv[2]) if len(sys.argv) > 2 else 50

df = pd.read_csv(path)
print("episodes: %d" % len(df))
print()

last = df.tail(window)
print("last %d episodes" % len(last))
print("  success rate        %.3f" % last.success.mean())
succ = last[last.success == 1]
if len(succ):
    print("  steps when success  %.1f" % succ.steps.mean())
print("  mean force          %.3f N" % last.force_mean.mean())
print("  p95 force           %.3f N" % last.force_p95.mean())
print("  max force           %.3f N" % last.force_max.max())
print("  steps over 0.85 N   %.1f per episode" % last.steps_over_threshold.mean())
print("  clamp steps         %.1f per episode" % last.clamp_steps.mean())
print("  end reasons         %s" % last.end_reason.value_counts().to_dict())
print()

print("success rate by block of %d" % window)
for i in range(0, len(df), window):
    block = df.iloc[i:i + window]
    if len(block) < window // 2:
        continue
    print("  ep %5d to %5d   success %.3f   force_mean %.3f   ep_return %+.3f"
          % (block.episode.iloc[0], block.episode.iloc[-1],
             block.success.mean(), block.force_mean.mean(), block.ep_return.mean()))

if "arch_type" in df:
    print()
    print("success by arch type, last %d" % len(last))
    for arch, grp in last.groupby("arch_type"):
        print("  %-14s n=%3d  success %.3f" % (arch, len(grp), grp.success.mean()))