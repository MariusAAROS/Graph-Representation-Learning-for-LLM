"""Difficulty scoring for curriculum learning.

Two complementary signals are combined into a single per-sample difficulty:

1. Task-level difficulty from out-of-distribution (OOD) transfer results. Each
   leave-one-task-out (LOTO) run logs ``val/exact_match_{task}`` for the single
   held-out task; a task the model generalizes to poorly is intrinsically
   harder. These are read from the local ``wandb`` run summaries.
2. Per-sample graph complexity (node count, edge count, density). Larger and
   denser graphs are harder to reason over. This is a *within-task* modifier so
   it does not compete with the cross-task OOD signal.

The two are blended as ``alpha * task_diff + (1 - alpha) * graph_diff`` and then
converted to a rank-based CDF in (0, 1], which the competence pacing function
consumes directly.
"""

import ast
import glob
import json
import os
import re

import numpy as np
import yaml

# Match per-task metric keys like "val/exact_match_TriangleCounting" while
# excluding the global "val/exact_match" (no task suffix).
_EXACT_MATCH_RE = re.compile(r"^val/exact_match_(?P<task>[A-Za-z][A-Za-z0-9]*)$")
_BASELINE_RE = re.compile(r"^val/random_baseline_(?P<task>[A-Za-z][A-Za-z0-9]*)$")


def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _read_run_config(files_dir):
    """Load a wandb run's ``config.yaml`` into a plain dict.

    wandb stores each hyperparameter as ``{key: {"value": <stringified repr>}}``;
    the inner value is a Python-literal string (single quotes, ``None``), so it
    is unwrapped with ``ast.literal_eval``. Returns ``{}`` if unavailable.
    """
    path = os.path.join(files_dir, "config.yaml")
    if not os.path.exists(path):
        return {}
    try:
        raw = yaml.safe_load(open(path)) or {}
    except (yaml.YAMLError, OSError):
        return {}
    out = {}
    for key, wrapped in raw.items():
        value = wrapped.get("value") if isinstance(wrapped, dict) else wrapped
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        out[key] = value
    return out


def _reconstruct_run_name(files_dir):
    """Rebuild a LOTO run's display name (``{logger.name}-ood-{ood_task}``).

    Mirrors how the runners name leave-one-task-out runs, so runs can be
    filtered by name without parsing the binary ``.wandb`` log. Training runs
    keep ``logger.name == 'meta-icl'`` while raw/inference ablations become
    ``meta-icl-raw``. Returns ``None`` if the name cannot be reconstructed.
    """
    cfg = _read_run_config(files_dir)
    logger = cfg.get("logger") or {}
    dataset = cfg.get("dataset") or {}
    name = logger.get("name") if isinstance(logger, dict) else None
    ood_task = dataset.get("ood_task") if isinstance(dataset, dict) else None
    if not name or not ood_task:
        return None
    return f"{name}-ood-{ood_task}"


def parse_ood_task_metrics(wandb_dir="wandb", run_name_prefix=None):
    """Aggregate per-task OOD exact-match / baseline from local wandb summaries.

    A run is treated as a leave-one-task-out (LOTO) OOD run when its summary
    contains exactly one ``val/exact_match_{task}`` key: the validation set of a
    LOTO run holds only the single held-out task, whereas a standard (in-domain)
    run logs a per-task key for every task. This lets us pick out OOD transfer
    scores without relying on run metadata.

    ``run_name_prefix`` optionally restricts to runs whose reconstructed display
    name starts with the given prefix (e.g. ``"meta-icl-ood-"`` selects the
    trained meta-ICL LOTO runs and excludes ``baseline`` / ``*-raw`` ablations).

    Returns a dict: ``{task: {"exact_match": mean, "baseline": mean, "n": count}}``.
    """
    summary_paths = sorted(
        glob.glob(os.path.join(wandb_dir, "*", "files", "wandb-summary.json"))
    )

    agg: dict = {}
    for path in summary_paths:
        try:
            summary = _read_json(path)
        except (json.JSONDecodeError, OSError):
            continue

        exact_by_task = {}
        baseline_by_task = {}
        for key, value in summary.items():
            m = _EXACT_MATCH_RE.match(key)
            if m and isinstance(value, (int, float)):
                exact_by_task[m.group("task")] = float(value)
                continue
            b = _BASELINE_RE.match(key)
            if b and isinstance(value, (int, float)):
                baseline_by_task[b.group("task")] = float(value)

        # LOTO heuristic: exactly one held-out task in this run's validation set.
        if len(exact_by_task) != 1:
            continue

        # Optional filter on the reconstructed run display name.
        if run_name_prefix is not None:
            run_name = _reconstruct_run_name(os.path.dirname(path))
            if run_name is None or not run_name.startswith(run_name_prefix):
                continue

        task, exact = next(iter(exact_by_task.items()))
        baseline = baseline_by_task.get(task, 0.0)
        bucket = agg.setdefault(task, {"exact": [], "baseline": []})
        bucket["exact"].append(exact)
        bucket["baseline"].append(baseline)

    metrics = {}
    for task, bucket in agg.items():
        metrics[task] = {
            "exact_match": float(np.mean(bucket["exact"])),
            "baseline": float(np.mean(bucket["baseline"])),
            "n": len(bucket["exact"]),
        }
    return metrics


def _minmax_normalize(values):
    """Min-max scale a 1-D array to [0, 1]; return zeros if constant."""
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def compute_task_difficulty(metrics, metric="gap"):
    """Turn aggregated OOD metrics into normalized per-task difficulty in [0, 1].

    - ``metric="raw"``: difficulty = 1 - exact_match. Simple, but sensitive to
      raw/untrained runs where every task scores near zero.
    - ``metric="gap"``: difficulty = 1 - clip(exact_match - baseline, 0, 1).
      Measures how far above the majority-class chance level the model reached;
      robust when absolute accuracies are low. This is the default.

    Difficulties are min-max normalized across tasks so the hardest observed
    task is 1.0 and the easiest is 0.0, matching the graph-difficulty scale.
    """
    if not metrics:
        return {}

    tasks = sorted(metrics)
    if metric == "raw":
        raw = [1.0 - metrics[t]["exact_match"] for t in tasks]
    elif metric == "gap":
        raw = [
            1.0 - float(np.clip(metrics[t]["exact_match"] - metrics[t]["baseline"], 0.0, 1.0))
            for t in tasks
        ]
    else:
        raise ValueError(f"Unknown ood_difficulty_metric: {metric}")

    normed = _minmax_normalize(raw)
    return {task: float(normed[i]) for i, task in enumerate(tasks)}


def load_or_compute_task_difficulty(
    cache_path, wandb_dir="wandb", metric="gap", force=False, run_name_prefix=None
):
    """Load cached per-task difficulty, or compute it from wandb and cache it.

    The cache is a JSON file mapping task name to difficulty in [0, 1]. Delete
    the file or pass ``force=True`` to recompute after new OOD runs.
    ``run_name_prefix`` restricts which LOTO runs contribute (see
    :func:`parse_ood_task_metrics`).
    """
    if cache_path and os.path.exists(cache_path) and not force:
        return _read_json(cache_path)

    metrics = parse_ood_task_metrics(wandb_dir, run_name_prefix=run_name_prefix)
    difficulty = compute_task_difficulty(metrics, metric=metric)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(difficulty, f, indent=2, sort_keys=True)
    return difficulty


def _graph_feature(record, feature):
    """Extract a numeric graph feature from a record (fields may be strings)."""
    nnodes = float(record.get("nnodes", 0) or 0)
    nedges = float(record.get("nedges", 0) or 0)
    if feature == "nnodes":
        return nnodes
    if feature == "nedges":
        return nedges
    if feature == "density":
        if nnodes < 2:
            return 0.0
        return 2.0 * nedges / (nnodes * (nnodes - 1.0))
    raise ValueError(f"Unknown graph feature: {feature}")


def graph_difficulty(records, features=("nnodes", "nedges", "density")):
    """Per-sample graph difficulty in [0, 1], normalized *within each task*.

    Each feature is min-max scaled across the samples of the same task, then the
    scaled features are averaged. Normalizing per task keeps this a within-task
    modifier (e.g. "a big triangle-counting graph vs. a small one") rather than
    letting raw graph size leak into the cross-task ordering, which the OOD
    signal already governs.
    """
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    feature_matrix = np.zeros((n, len(features)), dtype=np.float64)
    for j, feature in enumerate(features):
        feature_matrix[:, j] = [_graph_feature(r, feature) for r in records]

    tasks = [r.get("task", "unknown") for r in records]
    task_to_indices: dict = {}
    for i, task in enumerate(tasks):
        task_to_indices.setdefault(task, []).append(i)

    scaled = np.zeros_like(feature_matrix)
    for indices in task_to_indices.values():
        idx = np.asarray(indices)
        for j in range(len(features)):
            scaled[idx, j] = _minmax_normalize(feature_matrix[idx, j])

    return scaled.mean(axis=1)


def combined_sample_difficulty(
    records,
    task_difficulty,
    alpha=0.7,
    features=("nnodes", "nedges", "density"),
):
    """Blend task-level (OOD) and per-sample (graph) difficulty into [0, 1].

    ``alpha`` weights the task-level term; ``1 - alpha`` weights graph
    complexity. Tasks missing from ``task_difficulty`` (e.g. no OOD run yet)
    fall back to the neutral value 0.5.
    """
    n = len(records)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    task_term = np.array(
        [task_difficulty.get(r.get("task", "unknown"), 0.5) for r in records],
        dtype=np.float64,
    )
    graph_term = graph_difficulty(records, features=features)
    return alpha * task_term + (1.0 - alpha) * graph_term


def difficulty_to_cdf(difficulties):
    """Convert raw difficulties to a rank-based CDF in (0, 1].

    Mapping to cumulative density (fraction of samples at least as easy) makes
    the competence value directly interpretable as "the fraction of the dataset
    currently unlocked", following Platanios et al. (2019). Ties share the same
    CDF value via average ranking.
    """
    arr = np.asarray(difficulties, dtype=np.float64)
    n = arr.size
    if n == 0:
        return arr

    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    # Average-rank for ties so equally-difficult samples unlock together.
    sorted_vals = arr[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    # (rank + 1) / n gives values in (0, 1], with the hardest sample at 1.0.
    return (ranks + 1.0) / n
