
from __future__ import annotations
import os, json, csv, datetime
from typing import Dict, Any, List
import numpy as np

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def bench_short(bench: str) -> str:
    bench = (bench or "").lower()
    if bench.startswith("cartpole"): return "cp"
    if bench.startswith("atari"): return "at"
    return bench[:3] or "run"

def build_run_dir(runs_root, bench: str, strategy: str, tag: str = "") -> str:
    """
    Create a flat, searchable run directory directly under `runs_root`.

    New format:
        <runs_root>/<bench>__<strategy>__<tag>__rNNN__YYYYMMDD-HHMMSS

    Examples:
        runs/panda__tsn_improved_reuse__panda3__dm128_L3_H1_K20_drop0.10__s1__rsm-action_athr0.65__r003__20260410-153522
        runs/atari__tsn_origin_reuse__specs_atari_cl_5_minari_like_breakout_first__dm128_L3_H4_K20_drop0.10__s0__rsm-old_mem256_kl0.25_mcinf__r001__20260410-153540

    Notes:
      - everything is saved on ONE LEVEL directly under `runs_root`,
      - benchmark, method and params are visible in the directory name,
      - run number is appended for easier repeated-run tracking,
      - timestamp is kept at the end for uniqueness and chronological sorting.
    """
    import re
    from datetime import datetime
    from pathlib import Path

    def _slugify(x) -> str:
        s = str(x).strip()
        # replace whitespace with "-"
        s = re.sub(r"\s+", "-", s)
        # keep alnum + a few safe separators used in tags
        s = re.sub(r"[^A-Za-z0-9._=+\-]+", "-", s)
        # collapse duplicate "-"
        s = re.sub(r"-{2,}", "-", s)
        # trim noisy separators from ends
        s = s.strip("-._")
        return s or "run"

    root = Path(runs_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    parts = [_slugify(bench), _slugify(strategy)]
    if tag:
        parts.append(_slugify(tag))

    prefix = "__".join(parts)

    # find next run id for this exact prefix
    marker = prefix + "__r"
    run_ids = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(marker):
            continue
        tail = name[len(marker):]
        m = re.match(r"(\d+)", tail)
        if m:
            run_ids.append(int(m.group(1)))

    next_run_id = max(run_ids, default=0) + 1
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_name = f"{prefix}__r{next_run_id:03d}__{ts}"
    run_dir = root / run_name

    # just in case of very rare collision
    while run_dir.exists():
        next_run_id += 1
        run_name = f"{prefix}__r{next_run_id:03d}__{ts}"
        run_dir = root / run_name

    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)



def save_json(path: str, payload) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_task_gen_json(run_dir: str, step: int, payload) -> str:
    gen_dir = os.path.join(run_dir, "gen")
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f"task_{int(step)}.json")
    save_json(path, payload)
    return path


def save_matrix_csv(path: str, task_names: List[str], P: np.ndarray) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["after_task \\ on_task"] + task_names)
        for i, row in enumerate(P):
            w.writerow([task_names[i]] + [f"{v:.6f}" for v in row])

