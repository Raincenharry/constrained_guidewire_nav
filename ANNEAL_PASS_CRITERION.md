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

## Correction to the abort rule, same evening

The 1.7 threshold stands. The read epochs above were wrong.

Absolute steps are constant at 13 while d shrinks, so the relative squeeze accelerates
to the end. If the policy tracks with one epoch of lag, B = d(e-1)/d(e):

  e21 1.10  e22 1.10  e23 1.12  e24 1.13  e25 1.15
  e26 1.18  e27 1.22  e28 1.28  e29 1.39  e30 1.65

One epoch of lag peaks at 1.65, just under threshold. Two epochs of lag gives 1.36 at
e26 and 1.57 at e28, both passing, then 1.79 at e29 and 2.30 at e30. Reading at 26 and
28 would miss exactly the failure mode that matters.

REVISED READ EPOCHS: 27, 29, 31. Abort if B > 1.7 at any epoch, or if B exceeds the
reference curve above by more than 0.3, which is the growing lag signal that precedes it.

REVISED RESUBMIT: --anneal_from 100 --anneal_end_epoch 40, not end_epoch alone.
Seeds are running ep_cost 87, 132, 105 and the finished d150 slack seed sat at 98.5,
so d does not cross the operating cost until about epoch 22 to 25. The first half of
the nominal ten epoch descent does nothing. Starting at 100 puts the whole window in
the productive region, giving about 18 working epochs instead of 6.
