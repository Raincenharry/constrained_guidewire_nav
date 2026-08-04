"""Anneal arm, read from recovered training traces and episode logs.
Writes reads/2026_08_04_anneal.txt. Training only: no epoch 50 checkpoint,
no evaluation, so nothing here is comparable to the evaluated conditions."""
import glob, json, os
import pandas as pd

OUT = "reads/2026_08_04_anneal.txt"
A_FROM, A_TO, A_START, A_END = 150.0, 20.0, 20, 30
STEPS_PER_EPOCH = 6000

lines = []
def p(s=""):
    print(s)
    lines.append(str(s))

def dsched(e):
    if e <= A_START:
        return A_FROM
    if e >= A_END:
        return A_TO
    frac = (e - A_START) / float(A_END - A_START)
    return A_FROM + frac * (A_TO - A_FROM)

p("ANNEAL ARM. Training traces and episode logs only.")
p("No epoch 50 checkpoint and no evaluation: the array was cancelled.")
p("Schedule assumed from 27 July notes: d held at %.0f through epoch %d,"
  % (A_FROM, A_START))
p("linear to %.0f by epoch %d, held thereafter. Config check printed below."
  % (A_TO, A_END))
p("Source traces_2026_08_03.tar.gz and data/. Script anneal_read.py")
p("")

runs = sorted(glob.glob("traces/**/runs_sacpid_tip_anneal_*/**/progress.csv",
                        recursive=True))
for f in runs:
    run = [q for q in f.split(os.sep) if q.startswith("runs_")][0]
    d = pd.read_csv(f)
    cfg_path = os.path.join(os.path.dirname(f), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    flat = json.dumps(cfg)

    p("=" * 70)
    p(run)
    for key in ("anneal", "cost_limit", "warmup"):
        p("  config mentions '%s': %s" % (key, key in flat))

    ecol = next((c for c in ("Train/Epoch", "Metrics/Epoch", "epoch")
                 if c in d.columns), None)
    ccol = next((c for c in d.columns if c.endswith("EpCost")), None)
    lcol = next((c for c in d.columns if "Lagrange" in c or "Multiplier" in c),
                None)
    p("  columns: epoch=%s cost=%s lambda=%s  nrows=%d"
      % (ecol, ccol, lcol, len(d)))

    ep = d[ecol].astype(int).tolist() if ecol else list(range(len(d)))
    cost = d[ccol].tolist() if ccol else [float("nan")] * len(d)
    lam = d[lcol].tolist() if lcol else [float("nan")] * len(d)

    p("")
    p("  B_lag1 = ep_cost(e-1)/d(e), the 27 July convention.")
    p("  B_same = ep_cost(e)/d(e). Both printed: the notes and this script")
    p("  differed by one epoch of alignment. Verdict is identical either way.")
    p("  %5s %8s %10s %11s %8s %8s"
      % ("epoch", "d", "ep_cost", "lambda", "B_lag1", "B_same"))
    for i, e in enumerate(ep):
        dv = dsched(int(e))
        B1 = (cost[i - 1] / dv) if (i > 0 and dv > 0) else float("nan")
        B0 = (cost[i] / dv) if dv > 0 else float("nan")
        p("  %5d %8.2f %10.2f %11.6f %8.2f %8.2f"
          % (int(e), dv, cost[i], lam[i], B1, B0))
    post = [i for i, e in enumerate(ep) if int(e) >= A_START]
    if post:
        p("")
        p("  peak lambda from epoch %d: %.6f"
          % (A_START, max(lam[i] for i in post)))
    p("  last logged epoch %d, ep_cost %.2f" % (int(ep[-1]), cost[-1]))
    p("")

p("=" * 70)
p("BLOCK LEVEL, from episode logs. 100 episode blocks, dominant env_instance.")
p("")
for f in sorted(glob.glob("data/runs_sacpid_tip_anneal_*/episodes_seed*.csv")):
    run = f.split(os.sep)[1]
    d = pd.read_csv(f)
    if "env_instance" in d.columns:
        d = d[d.env_instance == d.env_instance.value_counts().idxmax()]
    d = d.reset_index(drop=True)
    ccol = next((c for c in ("shadow_tip", "hinge_cost_shadow", "ep_cost")
                 if c in d.columns), None)
    p(run + "   n=%d   cost column %s" % (len(d), ccol))
    p("  %6s %7s %8s %9s %9s %9s"
      % ("block", "epoch", "stuck", "inserted", "cost", "success"))
    cum = d.steps.cumsum() if "steps" in d.columns else None
    for i in range(0, len(d), 100):
        b = d.iloc[i:i + 100]
        if len(b) < 50:
            continue
        epoch = (cum.iloc[min(i + 99, len(d) - 1)] / STEPS_PER_EPOCH
                 if cum is not None else float("nan"))
        stuck = (b.inserted_final < 50).mean()
        p("  %6d %7.1f %8.2f %9.1f %9.2f %9.3f"
          % (i // 100, epoch, stuck, b.inserted_final.mean(),
             b[ccol].median() if ccol else float("nan"), b.success.mean()))
    p("")

os.makedirs("reads", exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote %s" % OUT)
