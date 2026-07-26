"""
Evaluate a trained OmniSafe policy on held out anatomies.

Loads the actor from an OmniSafe checkpoint, runs deterministic episodes
through SteveCMDP, and writes the same episode CSV format the training
runs produce, so analyse.py works on the output unchanged.

Default pool is the held out arch types (VI, VII) that no training run
has ever seen. Pass --pool train to evaluate on the training morphologies
instead, which gives the seen against unseen comparison.

Usage:
  python evaluate.py --checkpoint path/to/epoch-50.pt --episodes 100
  python evaluate.py --checkpoint path/to/epoch-50.pt --pool train --episodes 100
"""

import argparse
import json
import os
import sys

# Must run BEFORE steve_cmdp is imported, because EPISODE_LOG is read at
# module import time. Setting it later has no effect and the log stays empty.
def _preset_episode_log() -> str:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pool", default="test")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--stochastic", action="store_true")
    known, _ = ap.parse_known_args()

    tag = known.tag
    if not tag and known.checkpoint:
        # Derive a readable tag from the checkpoint path, e.g.
        # runs_r1_baseline_s0/SAC-{...}/seed-000-.../torch_save/epoch-50.pt
        parts = known.checkpoint.split(os.sep)
        run = next((p for p in parts if p.startswith("runs_")), "run")
        epoch = os.path.splitext(os.path.basename(known.checkpoint))[0]
        tag = "%s_%s" % (run, epoch)
    if not tag:
        tag = "eval"
    if known.stochastic:
        tag = tag + "_stoch"

    path = os.path.abspath(
        "./evals/%s_%s_seed%d.csv" % (tag, known.pool, known.seed)
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.environ["STEVE_EPISODE_LOG"] = path
    return path

EPISODE_LOG_PATH = _preset_episode_log()

import numpy as np
import torch

from omnisafe.models.actor.actor_builder import ActorBuilder

from steve_cmdp import SteveCMDP, ZeroCost, MaxAlongDeviceHinge


def find_config(checkpoint_path: str) -> dict:
    """OmniSafe writes config.json two directories above torch_save."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("config.json not found next to checkpoint: %s" % cfg_path)
    with open(cfg_path) as fh:
        return json.load(fh)


def build_actor(env: SteveCMDP, cfg: dict) -> torch.nn.Module:
    """Rebuild the actor with the architecture the run was trained with.

    Reading the sizes from config.json rather than hardcoding them means a
    checkpoint from a differently configured run still loads correctly.
    """
    model_cfgs = cfg.get("model_cfgs", {})
    actor_cfgs = model_cfgs.get("actor", {})
    hidden_sizes = actor_cfgs.get("hidden_sizes", [256, 256])
    activation = actor_cfgs.get("activation", "relu")
    init_mode = model_cfgs.get("weight_initialization_mode", "kaiming_uniform")
    actor_type = model_cfgs.get("actor_type", "gaussian_sac")

    builder = ActorBuilder(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_sizes=hidden_sizes,
        activation=activation,
        weight_initialization_mode=init_mode,
    )
    actor = builder.build_actor(actor_type)
    return actor


def load_actor_weights(actor: torch.nn.Module, checkpoint_path: str) -> None:
    """The OmniSafe checkpoint is {'pi': state_dict}. Only the actor is
    saved, no critics, no optimiser state, no replay buffer. That is enough
    for evaluation but means training cannot be resumed from it."""
    blob = torch.load(checkpoint_path, map_location="cpu")
    if "pi" not in blob:
        raise SystemExit(
            "checkpoint has no 'pi' key, found: %s" % list(blob.keys())
        )
    missing, unexpected = actor.load_state_dict(blob["pi"], strict=False)
    if missing:
        print("WARNING: missing keys when loading actor: %s" % missing)
    if unexpected:
        print("WARNING: unexpected keys when loading actor: %s" % unexpected)
    actor.eval()


def policy_action(actor: torch.nn.Module, obs: torch.Tensor,
                  deterministic: bool = True) -> torch.Tensor:
    """Mean action when deterministic, otherwise a sample from the policy.

    Deterministic is the default and the headline protocol, matching what the
    reference paper reports. Stochastic exists because the mean action is not
    the policy: a distribution whose mean sits between two branch choices
    commits to neither, while sampled actions resolve it. Arch type II fails
    every deterministic episode and succeeds during training at the rate of
    type I, which is that effect. The constrained policies show it more
    strongly than the unconstrained ones, so both are measured.
    """
    with torch.no_grad():
        try:
            return actor.predict(obs, deterministic=deterministic)
        except (AttributeError, TypeError):
            # Fallback for actor classes without predict()
            dist = actor(obs)
            return dist.mean if deterministic else dist.sample()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--pool", type=str, default="test", choices=["train", "test"],
                   help="test uses held out arch types VI and VII")
    p.add_argument("--seed", type=int, default=100,
                   help="drives the evaluation anatomy and target sequence")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--cost", type=str, default="zero", choices=["zero", "hinge"],
                   help="hinge logs a live ep_cost as well as the shadow cost")
    p.add_argument("--stochastic", action="store_true",
                   help="sample from the policy instead of taking the mean "
                        "action. Off by default, so deterministic stays the "
                        "headline protocol and every existing eval CSV "
                        "remains comparable.")
    args = p.parse_args()

    # env.set_seed drives the anatomy and target sequence but not torch, so a
    # stochastic run would be irreproducible without this. Deterministic runs
    # are unaffected, they never draw from the generator.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.checkpoint):
        raise SystemExit("checkpoint not found: %s" % args.checkpoint)

    cfg = find_config(args.checkpoint)
    cost_fn = ZeroCost() if args.cost == "zero" else MaxAlongDeviceHinge()

    print("=" * 60)
    print("evaluating %s" % args.checkpoint)
    print("pool: %s   episodes: %d   seed: %d   mode: %s"
          % (args.pool, args.episodes, args.seed,
             "stochastic" if args.stochastic else "deterministic"))
    print("trained with seed %s" % cfg.get("seed"))
    print("episode log: %s" % EPISODE_LOG_PATH)
    print("=" * 60)
    sys.stdout.flush()

    env = SteveCMDP(
        "SteveNav-v0",
        seed=args.seed,
        cost_fn=cost_fn,
        train=(args.pool == "train"),
        vary_anatomy=True,
    )
    # SteveCMDP builds with the constructor seed, but reseed explicitly so
    # the evaluation sequence is identical regardless of construction order.
    env.set_seed(args.seed)

    # OmniSafe wraps the env with ActionScale during training, which maps the
    # actor's tanh output in [-1, 1] onto the real action bounds. evaluate.py
    # builds SteveCMDP directly with no wrappers, so that map has to be applied
    # by hand. Without it a full push of 1.0 reaches stEVE as 1 mm/s instead of
    # 40 mm/s and the wire inserts 26 mm in 200 steps instead of 250.
    act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    print("action bounds: %s to %s" % (env.action_space.low, env.action_space.high))
    sys.stdout.flush()

    def scale_action(act: torch.Tensor) -> torch.Tensor:
        act = act.clamp(-1.0, 1.0)
        return act_low + (act + 1.0) * 0.5 * (act_high - act_low)

    actor = build_actor(env, cfg)
    load_actor_weights(actor, args.checkpoint)

    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset()
        for _ in range(env.max_episode_steps):
            raw_action = policy_action(actor, obs,
                                       deterministic=not args.stochastic)
            action = scale_action(raw_action)
            obs, reward, cost, terminated, truncated, info = env.step(action)
            if bool(terminated) or bool(truncated):
                break
        if info.get("end_reason") == "target_reached":
            successes += 1
        if (ep + 1) % 10 == 0:
            print("  %d/%d episodes, success so far %.3f"
                  % (ep + 1, args.episodes, successes / (ep + 1)))
            sys.stdout.flush()

    env.close()
    print("done. success rate %.3f over %d episodes"
          % (successes / args.episodes, args.episodes))
    print("csv: %s" % EPISODE_LOG_PATH)


if __name__ == "__main__":
    main()
