# Annealing pass criterion, fixed 27 July before 36073208 landed

B is redefined for a moving budget: B_t = EpCost_t / d_t at every epoch.
Report max(B_t) over the run, not the single activation value.

PASS requires all four, on all three seeds unless stated:
1. max(B_t) < 1.7 on every seed.
2. tail(200) insertion retention R: mean across seeds >= 0.90, worst seed >= 0.85.
3. tail(200) EpCost within 20 percent of the final d=20.
4. eval mean insertion across 3 seeds >= 160 mm, worst seed >= 120 mm,
   against fixed d20 seed mean of 125.5 mm.

FAIL (triggers the section 9 negative framing):
- any seed with R < 0.7, or
- eval mean insertion < 125.5 mm (no improvement over fixed d20).

The specific prediction being tested: fixed d20 s1 evaluates 63.2 mm / 0.030,
fixed d30 s1 evaluates 22.2 mm / 0.000. Annealed to final d=20 those collapsed
seeds should land near d20 s0 at 187.8 mm, without costing s0 anything.

## Early abort rule, added 27 July after reading the schedule

Schedule is linear, d0=150 at warmup_epochs=20 down to d=20 at anneal_end_epoch=30.
The relative squeeze accelerates: the last two steps are 33->46 = 0.72 and 20->33 = 0.61,
so the harshest proportional cut lands where the constraint finally binds. The d150 slack
seed ran ep_cost 98.5, so d crosses actual cost near epoch 24 and the whole binding
transient happens in about six epochs.

ABORT: read B_t = EpCost_t / d_t at epochs 26 and 28. If B_t > 1.7 at either, kill the
array and resubmit with --anneal_end_epoch 40, halving the descent rate through the
binding region. Six epochs lost is cheap; discovering at epoch 50 that the schedule
reproduced the fixed d collapse is not.
