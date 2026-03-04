from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from metrics import MetricsTracker

Row = Dict[str, Any]

def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _filter_ok(rows: Sequence[Row]) -> List[Row]:
    return [r for r in rows if r.get("status") == "ok"]


def _group_rows(rows: Sequence[Row], keys: Tuple[str, ...]) -> Dict[Tuple[Any, ...], List[Row]]:
    groups: Dict[Tuple[Any, ...], List[Row]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        g = tuple(r.get(k) for k in keys)
        groups.setdefault(g, []).append(r)
    return groups


# # ----------------------------
# # Per-batch plots (MetricsTracker)
# # ----------------------------
# def plot_accuracy(tracker: MetricsTracker, save_path: Optional[str] = None) -> None:
#     if not tracker.top1_running:
#         print("No accuracy data to plot")
#         return

#     fig = plt.figure()
#     plt.plot(tracker.top1_running, label="Top-1")
#     plt.plot(tracker.top5_running, label="Top-5")
#     plt.xlabel("Batch")
#     plt.ylabel("Accuracy (%)")
#     plt.title("Eval Accuracy (running)")
#     plt.legend()
#     plt.tight_layout()

#     if save_path:
#         _ensure_parent_dir(save_path)
#         plt.savefig(save_path, dpi=200)

#     plt.show()
#     plt.close(fig)


# def plot_timing(tracker: MetricsTracker, save_path: Optional[str] = None) -> None:
#     if not tracker.infer_times_s:
#         print("No timing data to plot")
#         return

#     fig = plt.figure()
#     plt.plot([t * 1000 for t in tracker.infer_times_s], label="Infer (ms)")
#     plt.plot([t * 1000 for t in tracker.batch_times_s], label="Batch total (ms)")
#     plt.xlabel("Batch")
#     plt.ylabel("Time (ms)")
#     plt.title("Timing per batch")
#     plt.legend()
#     plt.tight_layout()

#     if save_path:
#         _ensure_parent_dir(save_path)
#         plt.savefig(save_path, dpi=200)

#     plt.show()
#     plt.close(fig)


# def plot_loss(tracker: MetricsTracker, save_path: Optional[str] = None) -> None:
#     if not tracker.losses:
#         print("No loss data to plot")
#         return

#     fig = plt.figure()
#     plt.plot(tracker.losses, label="Eval loss")
#     plt.xlabel("Batch")
#     plt.ylabel("Loss")
#     plt.title("Eval Loss")
#     plt.legend()
#     plt.tight_layout()

#     if save_path:
#         _ensure_parent_dir(save_path)
#         plt.savefig(save_path, dpi=200)

#     plt.show()
#     plt.close(fig)


# ----------------------------
# Thesis plots from flattened rows
# ----------------------------
def plot_metric_vs_input_bits(
    rows: Sequence[Row],
    metric_key: str,
    *,
    group_by: Tuple[str, str] = ("cfg.backend", "cfg.model_precision"),
    title: Optional[str] = None,
    connect_points: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """
    Discrete plot (scatter) of metric vs cfg.input_quant_bits.
    Expects flattened keys: "cfg.input_quant_bits" and metric_key like "res.top1_acc".
    """
    ok = _filter_ok(rows)
    groups = _group_rows(ok, group_by)

    fig = plt.figure()
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: _as_int(r.get("cfg.input_quant_bits"), 0))
        xs = [_as_int(r.get("cfg.input_quant_bits"), 0) for r in rs]
        ys = [_as_float(r.get(metric_key)) for r in rs]

        plt.scatter(xs, ys, label=f"{g[0]} | {g[1]}")
        if connect_points and len(xs) >= 2:
            plt.plot(xs, ys)

    plt.xlabel("input_quant_bits (0 = none)")
    plt.ylabel(metric_key)
    plt.title(title or f"{metric_key} vs input bits")
    plt.legend()
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close(fig)


def plot_tradeoff_scatter(
    rows: Sequence[Row],
    *,
    x_key: str = "res.infer_ms_avg",
    y_key: str = "res.top1_acc",
    label_keys: Tuple[str, ...] = ("cfg.backend", "cfg.model_precision", "cfg.input_quant_bits"),
    title: str = "Accuracy vs Latency",
    annotate: bool = False,
    save_path: Optional[str] = None,
) -> None:
    ok = _filter_ok(rows)

    xs, ys, labels = [], [], []
    for r in ok:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is None or y is None:
            continue
        xs.append(_as_float(x))
        ys.append(_as_float(y))
        labels.append(" | ".join(str(r.get(k)) for k in label_keys))

    fig = plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)

    if annotate:
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y), fontsize=8)

    plt.tight_layout()
    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close(fig)


def _pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    # minimize x (latency), maximize y (accuracy)
    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    frontier: List[Tuple[float, float]] = []
    best_y = -float("inf")
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot_tradeoff_with_pareto(
    rows: Sequence[Row],
    *,
    x_key: str = "res.infer_ms_avg",
    y_key: str = "res.top1_acc",
    title: str = "Accuracy–Latency tradeoff (Pareto)",
    save_path: Optional[str] = None,
) -> None:
    ok = _filter_ok(rows)

    pts: List[Tuple[float, float]] = []
    for r in ok:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is None or y is None:
            continue
        pts.append((_as_float(x), _as_float(y)))

    fig = plt.figure()
    plt.scatter([p[0] for p in pts], [p[1] for p in pts])

    frontier = _pareto_frontier(pts)
    if len(frontier) >= 2:
        plt.plot([p[0] for p in frontier], [p[1] for p in frontier])

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close(fig)


def plot_delta_from_baseline(
    rows: Sequence[Row],
    *,
    baseline_selector: Dict[str, Any],
    acc_key: str = "res.top1_acc",
    lat_key: str = "res.infer_ms_avg",
    group_by: Tuple[str, ...] = ("cfg.backend", "cfg.model_precision"),
    title: str = "Delta from baseline",
    save_path: Optional[str] = None,
) -> None:
    ok = _filter_ok(rows)

    baseline = None
    for r in ok:
        if all(r.get(k) == v for k, v in baseline_selector.items()):
            baseline = r
            break
    if baseline is None:
        raise ValueError(f"No baseline found for selector: {baseline_selector}")

    b_acc = _as_float(baseline.get(acc_key))
    b_lat = _as_float(baseline.get(lat_key))

    groups = _group_rows(ok, group_by)

    # ΔAcc plot
    fig = plt.figure()
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: _as_int(r.get("cfg.input_quant_bits"), 0))
        xs = [_as_int(r.get("cfg.input_quant_bits"), 0) for r in rs]
        dacc = [_as_float(r.get(acc_key)) - b_acc for r in rs]
        plt.scatter(xs, dacc, label=str(g))

    plt.axhline(0.0)
    plt.xlabel("input_quant_bits (0 = none)")
    plt.ylabel(f"Δ {acc_key}")
    plt.title(title + " — accuracy")
    plt.legend()
    plt.tight_layout()
    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close(fig)

    # Speedup plot (saved as *_speedup if save_path provided)
    fig2 = plt.figure()
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: _as_int(r.get("cfg.input_quant_bits"), 0))
        xs = [_as_int(r.get("cfg.input_quant_bits"), 0) for r in rs]
        sp = []
        for r in rs:
            lat = _as_float(r.get(lat_key))
            if lat <= 0 or math.isnan(lat) or math.isnan(b_lat):
                sp.append(float("nan"))
            else:
                sp.append(b_lat / lat)
        plt.scatter(xs, sp, label=str(g))

    plt.axhline(1.0)
    plt.xlabel("input_quant_bits (0 = none)")
    plt.ylabel("Speedup (baseline_lat / lat)")
    plt.title(title + " — speedup")
    plt.legend()
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        speed_path = p.with_name(p.stem + "_speedup" + p.suffix)
        _ensure_parent_dir(speed_path)
        plt.savefig(speed_path, dpi=200)

    plt.show()
    plt.close(fig2)