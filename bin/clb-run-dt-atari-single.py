#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch
import torch.nn.functional as F

import ale_py
import gymnasium as gym

from clbench.io.run_logger import build_run_dir, save_json

gym.register_envs(ale_py)

from dt.model import DecisionTransformer
from dt.utils import (
    make_minari_atari_env,
    load_npz_dataset_for_task,
    make_offline_minibatches,
    evaluate_dt_forward,
)

# -----------------------------
# Helpers
# -----------------------------
def _model_signature(args) -> str:
    return (
        f"dm{int(args.d_model)}"
        f"_L{int(args.n_layers)}"
        f"_H{int(args.n_heads)}"
        f"_K{int(args.seq_len)}"
        f"_drop{float(args.p_drop):.2f}"
    )


def _seed_signature(seed_override: Optional[int], seed_offset: int) -> str:
    if seed_override is not None:
        return f"__s{int(seed_override)}"
    if int(seed_offset) != 0:
        return f"__sspec_off{int(seed_offset)}"
    return "__sspec"


def _resolve_task_seed(task: dict, idx: int, seed_override: Optional[int], seed_offset: int) -> Tuple[int, int]:
    base_seed = int(task.get("seed", idx))
    effective_seed = int((seed_override if seed_override is not None else base_seed) + int(seed_offset))
    return base_seed, effective_seed

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def obs_diff_stats(obs_env, obs_ds) -> Tuple[float, float]:
    a = np.asarray(obs_env, dtype=np.float32)
    b = np.asarray(obs_ds, dtype=np.float32)

    if a.size and a.max() > 1.5:
        a = a / 255.0
    if b.size and b.max() > 1.5:
        b = b / 255.0

    diff = np.abs(a - b)
    return float(diff.max()), float(diff.mean())


def replay_actions(
    env: gym.Env,
    actions: np.ndarray,
    *,
    seed: int,
    max_steps: Optional[int] = None,
) -> Tuple[float, int, bool, bool, Optional[int]]:
    obs, info = env.reset(seed=int(seed))
    total = 0.0
    terminated = False
    truncated = False
    steps = 0
    last_info = info

    for t, a in enumerate(np.asarray(actions, dtype=np.int64)):
        if max_steps is not None and t >= int(max_steps):
            break
        obs, r, terminated, truncated, info = env.step(int(a))
        total += float(r)
        steps = t + 1
        last_info = info
        if terminated or truncated:
            break

    lives = None
    if isinstance(last_info, dict):
        lives = last_info.get("lives", None)
    return float(total), int(steps), bool(terminated), bool(truncated), lives


def infer_action_map_small_discrete(
    env: gym.Env,
    ep_actions: np.ndarray,
    ds_return: float,
    *,
    seed: int,
    tol: float = 1e-3,
    max_n: int = 6,
) -> Optional[np.ndarray]:
    if not isinstance(env.action_space, gym.spaces.Discrete):
        return None
    n = int(env.action_space.n)
    if n > max_n:
        return None

    ep_actions = np.asarray(ep_actions, dtype=np.int64)
    if ep_actions.size == 0:
        return None
    if ep_actions.min() < 0 or ep_actions.max() >= n:
        return None

    ret0, *_ = replay_actions(env, ep_actions, seed=seed)
    diff0 = abs(ret0 - float(ds_return))
    if diff0 <= tol:
        return None

    best_map = None
    best_diff = diff0

    for perm in itertools.permutations(range(n)):
        m = np.asarray(perm, dtype=np.int64)
        mapped = m[ep_actions]
        ret, *_ = replay_actions(env, mapped, seed=seed)
        d = abs(ret - float(ds_return))
        if d < best_diff:
            best_diff = d
            best_map = m
            if best_diff <= tol:
                break

    if best_map is not None and best_diff <= tol:
        return best_map
    return None


def pick_target_return(returns: np.ndarray, mode: str) -> float:
    mode = str(mode).lower()
    if returns.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(returns))
    if mode == "p90":
        return float(np.percentile(returns, 90))
    if mode == "mean":
        return float(np.mean(returns))
    raise ValueError("mode must be max|p90|mean")


def train_offline_dt(
    model: DecisionTransformer,
    episodes_obs,
    episodes_actions,
    episodes_rewards,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    seq_len: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
):
    """
    Train a Decision Transformer offline.

    IMPORTANT:
    This DT uses token order (R_t, s_t, a_t) and predicts actions from state tokens.
    Therefore, we should feed the *current* action tokens (unshifted).
    Causality ensures s_t cannot attend to a_t.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_offline_minibatches(
        episodes_obs, episodes_actions, episodes_rewards,
        seq_len=seq_len, batch_size=batch_size, device=device
    )

    model.train()
    for step in range(int(steps)):
        obs, actions, rtg, ts, mask = next(loader)

        if step == 0:
            with torch.no_grad():
                obs_f = obs.float()
                print(f"[train] batch obs dtype={obs.dtype}, raw_min={float(obs_f.min()):.3f}, raw_max={float(obs_f.max()):.3f}")
                print(f"[train] batch actions dtype={actions.dtype}, min={int(actions.min()):d}, max={int(actions.max()):d}")
                print(f"[train] batch rtg dtype={rtg.dtype}, min={float(rtg.min()):.3f}, max={float(rtg.max()):.3f}")
                print(f"[train] model.rtg_scale={getattr(model, 'rtg_scale', None)}")

        # ✅ feed actions as-is (may include -1 padding; model clamps to >=0 for embeddings)
        logits = model(obs, actions, rtg, ts, attention_mask=mask)  # [B,L,A]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            actions.reshape(-1),
            ignore_index=-1,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % max(1, int(steps) // 10) == 0:
            print(f"[train] step {step+1}/{steps}, loss={loss.item():.4f}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="runs")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed-override", type=int, default=None,
                    help="If set, overrides per-task seeds from spec.")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Offset added to the final effective seed.")
    ap.add_argument("--spec", required=True, help="JSON list of tasks (name, seed, params.game)")
    ap.add_argument("--dataset-root", default="resources/atari_expert")
    ap.add_argument("--dataset-file", default="expert_minari_dqn.npz")

    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=64)

    ap.add_argument("--episodes-eval", type=int, default=10)
    ap.add_argument("--max-ep-len", type=int, default=27000)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--p-drop", type=float, default=0.1)

    ap.add_argument("--min-episode-return", type=float, default=None)
    ap.add_argument("--target-mode", choices=["max", "p90", "mean"], default="max")
    ap.add_argument("--target-return", type=float, default=None)

    ap.add_argument("--debug-replay", action="store_true")
    ap.add_argument("--debug-episode", type=int, default=0)
    ap.add_argument("--debug-only", action="store_true")

    # Keep compatibility flags, but enable auto-fire only heuristically (Breakout)
    ap.add_argument("--auto-fire", action="store_true")
    ap.add_argument("--auto-fire-on-life-loss", action="store_true")

    ap.add_argument("--rtg-scale", type=float, default=0.0)
    ap.add_argument("--dqn-size", type=int, default=84)

    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[device] {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    print(f"[seed] mode={'override' if args.seed_override is not None else 'spec'}")
    print(f"[seed] seed_override={args.seed_override}")
    print(f"[seed] seed_offset={int(args.seed_offset)}")
    print(f"[determinism] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>')}")

    tasks = json.loads(open(args.spec, "r", encoding="utf-8").read())
    assert isinstance(tasks, list) and len(tasks) > 0

    spec_tag = os.path.splitext(os.path.basename(args.spec))[0]
    model_sig = _model_signature(args)
    run_tag = f"{(args.tag or spec_tag)}__{model_sig}{_seed_signature(args.seed_override, args.seed_offset)}"
    run_dir = build_run_dir(args.runs_root, "atari", strategy="single_task_dt", tag=run_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run_dir] {run_dir}")

    results: Dict[str, float] = {}
    task_records: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks, start=1):
        name = task["name"]
        base_seed, seed = _resolve_task_seed(task, i, args.seed_override, args.seed_offset)
        p = task.get("params", {})
        env_id = p["game"]

        frame_stack = int(p.get("frame_stack", 4))
        clip_rewards = bool(p.get("clip_rewards", True))

        print(f"\n[Single-task {i}/{len(tasks)}] {name} ({env_id})")
        print(f"[seed] task={name} base_seed={base_seed} effective_seed={seed}")

        set_all_seeds(seed)

        env = make_minari_atari_env(
            env_id=env_id,
            seed=seed,
            frame_stack=frame_stack,
            dqn_size=int(args.dqn_size),
            clip_rewards=clip_rewards,
        )

        npz_path = os.path.join(args.dataset_root, name, args.dataset_file)
        print(f"[data] loading {npz_path}")
        episodes_obs, episodes_actions, episodes_rewards, ep_returns = load_npz_dataset_for_task(npz_path)

        print(f"[data] episodes={len(episodes_obs)}, mean={ep_returns.mean():.1f}, min={ep_returns.min():.1f}, max={ep_returns.max():.1f}")
        sample = episodes_obs[0][0]
        print(f"[data] obs dtype={sample.dtype}, min={float(sample.min()):.3f}, max={float(sample.max()):.3f}, shape={sample.shape}")

        if args.min_episode_return is not None:
            thr = float(args.min_episode_return)
            keep = ep_returns >= thr
            episodes_obs = [ep for ep, k in zip(episodes_obs, keep) if k]
            episodes_actions = [ep for ep, k in zip(episodes_actions, keep) if k]
            episodes_rewards = [ep for ep, k in zip(episodes_rewards, keep) if k]
            ep_returns = ep_returns[keep]
            print(f"[data] filtered return>={thr}: kept {keep.sum()}/{len(keep)}")
            print(f"[data] new mean={ep_returns.mean():.1f} min={ep_returns.min():.1f} max={ep_returns.max():.1f}")

        if len(episodes_actions) == 0:
            raise RuntimeError("No episodes after filtering.")

        ds0 = float(np.sum(np.asarray(episodes_rewards[0], dtype=np.float32)))
        if args.rtg_scale > 0:
            rtg_scale = float(args.rtg_scale)
        else:
            rtg_scale = float(max(1.0, np.max(np.abs(ep_returns))))
        print(f"[data] rtg_scale used = {rtg_scale:.1f}")

        env0, *_ = replay_actions(env, np.asarray(episodes_actions[0], dtype=np.int64), seed=seed)
        if abs(env0 - ds0) <= 1e-3:
            print(f"[data] action mapping identity (ep0 replay matches dataset): {ds0:.2f} == {env0:.2f} (seed={seed})")
        if abs(env0 - ds0) <= 1e-3:
            print(f"[data] action mapping identity (ep0 replay matches dataset): {ds0:.2f} == {env0:.2f}")
        else:
            m = infer_action_map_small_discrete(env, np.asarray(episodes_actions[0]), ds0, seed=seed, tol=1e-3, max_n=6)
            if m is not None:
                episodes_actions = [m[np.asarray(ep, dtype=np.int64)] for ep in episodes_actions]
                env1, *_ = replay_actions(env, np.asarray(episodes_actions[0], dtype=np.int64), seed=seed)
                print(f"[data] action remap found ds->env = {m.tolist()}")
                print(f"[data] env meanings = {env.unwrapped.get_action_meanings()}")
                print(f"[data] after remap ep0 replay: {ds0:.2f} == {env1:.2f} (seed={seed})")
            else:
                print(f"[warn] ep0 replay mismatch: dataset={ds0:.2f} env={env0:.2f}")
                print("[warn] This usually means: env != dataset (preprocess/wrappers/frameskip/sticky/noop).")

        # ---- build model ----
        obs_shape = env.observation_space.shape
        n_actions = env.action_space.n

        max_len_in_data = max(len(r) for r in episodes_rewards) + 1
        max_ep_len_embed = max(10000, max_len_in_data, int(args.max_ep_len) + 1)

        model = DecisionTransformer(
            obs_shape=obs_shape,
            n_actions=n_actions,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            seq_len=args.seq_len,
            p_drop=args.p_drop,
            max_ep_len=max_ep_len_embed,
            rtg_scale=float(rtg_scale),
        ).to(device)

        # ---- train ----
        train_offline_dt(
            model=model,
            episodes_obs=episodes_obs,
            episodes_actions=episodes_actions,
            episodes_rewards=episodes_rewards,
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
            seq_len=args.seq_len,
        )

        # ---- eval (STABLE: forward, seeded) ----
        model.eval()
        if args.target_return is not None:
            target = float(args.target_return)
        else:
            target = pick_target_return(ep_returns, args.target_mode)

        try:
            meanings = env.unwrapped.get_action_meanings()
        except Exception:
            meanings = []

        # Auto-fire policy:
        # - default: follow CLI flags
        # - automatic fallback: enable only if the dataset NEVER uses FIRE
        auto_fire_eff = bool(args.auto_fire)
        auto_fire_life_eff = bool(args.auto_fire_on_life_loss)

        if "FIRE" in meanings:
            fire_id = meanings.index("FIRE")
            fire_used = any((np.asarray(ep, dtype=np.int64) == fire_id).any() for ep in episodes_actions)

            # If FIRE never appears in the dataset, the dataset was likely collected with an auto-fire helper.
            # In that case enabling auto-fire is beneficial and does not shift the distribution.
            if not fire_used:
                auto_fire_eff = True
                auto_fire_life_eff = True

        print(f"[eval] auto_fire={auto_fire_eff} auto_fire_on_life_loss={auto_fire_life_eff} eval_seed={seed}")

        avg_ret = evaluate_dt_forward(
            model=model,
            env=env,
            episodes=args.episodes_eval,
            device=device,
            max_steps=args.max_ep_len,
            target_return=target,
            seed=seed,                     # ✅ deterministic episodes: seed+ep
            greedy=True,                   # ✅ stable (argmax)
            clamp_to_env_actions=True,
            auto_fire=auto_fire_eff,
            auto_fire_on_life_loss=auto_fire_life_eff,
            debug_action_hist=False,
        )

        results[name] = float(avg_ret)
        print(f"[eval] {name}: avg_return={avg_ret:.3f} (target_return={target:.2f}, seed={seed})")
        task_rec = {
            "task": name,
            "env_id": env_id,
            "dataset_path": npz_path,
            "base_seed": int(base_seed),
            "effective_seed": int(seed),
            "score": float(avg_ret),
            "target_return": float(target),
            "target_mode": str(args.target_mode) if args.target_return is None else "manual",
            "rtg_scale": float(rtg_scale),
            "episodes_eval": int(args.episodes_eval),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "max_ep_len_eval": int(args.max_ep_len),
            "d_model": int(args.d_model),
            "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads),
            "p_drop": float(args.p_drop),
            "lr": float(3e-4),
            "weight_decay": float(1e-4),
            "frame_stack": int(frame_stack),
            "clip_rewards": bool(clip_rewards),
            "auto_fire": bool(auto_fire_eff),
            "auto_fire_on_life_loss": bool(auto_fire_life_eff),
        }
        task_records.append(task_rec)

        save_json(os.path.join(run_dir, f"{name}.json"), task_rec)

        avg_score = float(np.mean(list(results.values()))) if results else 0.0

        save_json(
            os.path.join(run_dir, "results.json"),
            {
                "mode": "single-task-atari",
                "spec": str(args.spec),
                "tasks_ran": list(results.keys()),
                "scores": results,
                "avg_score": avg_score,
                "seed_override": None if args.seed_override is None else int(args.seed_override),
                "seed_offset": int(args.seed_offset),
                "task_records": task_records,
                "device": str(device),
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "seq_len": int(args.seq_len),
                "max_ep_len": int(args.max_ep_len),
                "episodes_eval": int(args.episodes_eval),
                "target_mode": str(args.target_mode),
                "target_return_override": None if args.target_return is None else float(args.target_return),
                "d_model": int(args.d_model),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "p_drop": float(args.p_drop),
                "dataset_root": str(args.dataset_root),
                "dataset_file": str(args.dataset_file),
            },
        )

        try:
            env.close()
        except Exception:
            pass

    print("\n=== Single-task Minari Atari DT results ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
