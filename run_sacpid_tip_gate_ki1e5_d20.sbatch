#!/bin/bash
#SBATCH --job-name=tip_gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4
#SBATCH --output=slurm_logs/tipgate_%A_%a.out
#SBATCH --error=slurm_logs/tipgate_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Gated controller on the tip signal. Fixed d=20, warmup 20, eight seeds.
# Two differences from sacpid_tip_w20_d20, and they are not independent:
# the controller is called once per epoch instead of once per gradient step,
# and ki is raised from the OmniSafe default of 1e-7 to 1e-5 to compensate.
#
# Why both together. sac_pid.py lines 62 to 65 call pid_update from _update,
# which the off policy loop in ddpg.py runs once per gradient step, so at
# steps_per_epoch 6000 the controller fires about 6000 times per epoch. The
# published PID Lagrangian gains assume one call per epoch, which is true for
# on policy algorithms. Ported to an off policy algorithm the effective
# integral gain is inflated by the number of updates per epoch. The gate,
# committed as 533acca and verified on WSL, restores the intended call rate.
#
# ki is raised because the gate alone is a guaranteed null. Measured on WSL
# at steps_per_epoch 500, gate on, default ki 1e-7: lambda moved from
# 0.00100921 to 0.00105655 across nine epochs, about 5e-6 per epoch. That
# rate is ki times the error and does not depend on steps_per_epoch, so a
# 6000 step run behaves the same. The collapse boundary from six fixed budget
# runs is 0.009 survive, 0.024 break, so gated default never engages inside
# 30 post warmup epochs and would reproduce the d150 slack control.
#
# Gated, lambda grows by ki times the error per epoch. At an error near 70:
#   gated ki 5e-6   3.5e-4 per epoch   reaches 0.009 in about 23 epochs
#   gated ki 1e-5   7.0e-4 per epoch   reaches 0.009 in about 11 epochs
#   gated ki 6e-5   4.2e-3 per epoch   equals the existing ki 1e-8 runs
# 1e-5 chosen: slow enough for the policy to adapt, fast enough to bind
# inside the run. Eight seeds because the seed level permutation test cannot
# reach significance below eight.
#
# READ RULE, written before the data exists. Read LAG lines at epochs 25, 30,
# 35, 40.
#   If lambda climbs past 0.024 while ep_cost is still above 40, this is the
#   same failure on a slower clock and d=20 is infeasible at any gain. That
#   is the answer, not a wasted run.
#   If lambda settles below 0.009 and ep_cost falls toward 20 from above
#   without undershooting to near zero, the constraint is regulating.
#   If lambda stays under 0.002 through epoch 40, ki 1e-5 is still too small
#   and the finding is that the feasible gain window on this task is narrow.
#
# ISOLATION. Nothing here writes into evals/, traces/, reads/ or figs_31_july/.
# Evaluations of these runs go to evals_gate/, never evals/, until after the
# report is submitted on 27 August.
SEED=${SLURM_ARRAY_TASK_ID}
TAG="sacpid_tip_gate_ki1e5_d20_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> fixed d20 seed ${SEED} warmup 20 gate on ki 1e-5 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit 20 \
  --warmup_epochs 20 \
  --pid_gate_per_epoch \
  --pid_ki 1e-5 \
  --pid_log_calls \
  --tag ${TAG}
echo "end: $(date)"
