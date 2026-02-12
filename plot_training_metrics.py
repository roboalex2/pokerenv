from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _to_float(value: str) -> float:
    if value is None:
        return float("nan")
    value = value.strip()
    if value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def _to_int(value: str) -> int:
    if value is None:
        return 0
    value = value.strip()
    if value == "":
        return 0
    return int(float(value))


def load_metrics(path: str) -> Dict[str, List[float]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in metrics file: {path}")

    data: Dict[str, List[float]] = {
        "iteration": [],
        "time_total_s": [],
        "time_traversals_s": [],
        "time_adv_train_s": [],
        "time_strategy_train_s": [],
        "adv_loss_mean": [],
        "strategy_loss": [],
        "regret_buffer_total_size": [],
        "strategy_buffer_size": [],
        "regret_samples_added_total": [],
        "strategy_samples_added": [],
        "periodic_checkpoint_saved": [],
    }

    for row in rows:
        data["iteration"].append(_to_int(row.get("iteration", "0")))
        data["time_total_s"].append(_to_float(row.get("time_total_s", "nan")))
        data["time_traversals_s"].append(_to_float(row.get("time_traversals_s", "nan")))
        data["time_adv_train_s"].append(_to_float(row.get("time_adv_train_s", "nan")))
        data["time_strategy_train_s"].append(_to_float(row.get("time_strategy_train_s", "nan")))
        data["adv_loss_mean"].append(_to_float(row.get("adv_loss_mean", "nan")))
        data["strategy_loss"].append(_to_float(row.get("strategy_loss", "nan")))
        data["regret_buffer_total_size"].append(_to_float(row.get("regret_buffer_total_size", "nan")))
        data["strategy_buffer_size"].append(_to_float(row.get("strategy_buffer_size", "nan")))
        data["regret_samples_added_total"].append(_to_float(row.get("regret_samples_added_total", "nan")))
        data["strategy_samples_added"].append(_to_float(row.get("strategy_samples_added", "nan")))
        data["periodic_checkpoint_saved"].append(_to_int(row.get("periodic_checkpoint_saved", "0")))

    return data


def _has_finite(values: List[float]) -> bool:
    return any(math.isfinite(v) for v in values)


def plot_metrics(data: Dict[str, List[float]], title: str, output_path: str) -> None:
    x = data["iteration"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    # 1) Time breakdown.
    axes[0].plot(x, data["time_total_s"], label="total", linewidth=1.8)
    axes[0].plot(x, data["time_traversals_s"], label="traversals", linewidth=1.5)
    axes[0].plot(x, data["time_adv_train_s"], label="adv_train", linewidth=1.5)
    axes[0].plot(x, data["time_strategy_train_s"], label="strategy_train", linewidth=1.5)
    axes[0].set_title("Iteration Time (s)")
    axes[0].set_ylabel("seconds")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.25)

    # 2) Loss curves.
    if _has_finite(data["adv_loss_mean"]):
        axes[1].plot(x, data["adv_loss_mean"], label="adv_loss_mean", linewidth=1.6)
    if _has_finite(data["strategy_loss"]):
        axes[1].plot(x, data["strategy_loss"], label="strategy_loss", linewidth=1.6)
    axes[1].set_title("Training Loss")
    axes[1].set_ylabel("loss")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.25)

    # 3) Buffer sizes.
    axes[2].plot(x, data["regret_buffer_total_size"], label="regret_buffer_total", linewidth=1.8)
    axes[2].plot(x, data["strategy_buffer_size"], label="strategy_buffer", linewidth=1.8)
    axes[2].set_title("Buffer Sizes")
    axes[2].set_ylabel("samples")
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].grid(alpha=0.25)

    # 4) Samples added per iteration.
    axes[3].plot(x, data["regret_samples_added_total"], label="regret_added", linewidth=1.6)
    axes[3].plot(x, data["strategy_samples_added"], label="strategy_added", linewidth=1.6)
    axes[3].set_title("Samples Added Per Iteration")
    axes[3].set_ylabel("samples")
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(alpha=0.25)

    # 5) Traversal share of total time.
    share = []
    for t_trav, t_tot in zip(data["time_traversals_s"], data["time_total_s"]):
        if t_tot > 0 and math.isfinite(t_trav) and math.isfinite(t_tot):
            share.append(100.0 * t_trav / t_tot)
        else:
            share.append(float("nan"))
    axes[4].plot(x, share, label="traversal_share", linewidth=1.6)
    axes[4].set_title("Traversal Share of Iteration Time")
    axes[4].set_ylabel("%")
    axes[4].legend(loc="upper right", fontsize=9)
    axes[4].grid(alpha=0.25)

    # 6) Checkpoint markers.
    ckpt_iters = [it for it, s in zip(x, data["periodic_checkpoint_saved"]) if s == 1]
    axes[5].plot(x, data["periodic_checkpoint_saved"], linewidth=1.2, label="checkpoint_saved")
    if ckpt_iters:
        axes[5].scatter(ckpt_iters, [1] * len(ckpt_iters), s=20, label="checkpoint points")
    axes[5].set_title("Checkpoint Events")
    axes[5].set_ylabel("flag")
    axes[5].set_yticks([0, 1])
    axes[5].legend(loc="upper right", fontsize=9)
    axes[5].grid(alpha=0.25)

    for ax in axes:
        ax.set_xlabel("iteration")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PNG charts from training_metrics.csv")
    parser.add_argument("--metrics-file", type=str, default="training_metrics.csv", help="Input metrics CSV path")
    parser.add_argument("--output", type=str, default="training_metrics.png", help="Output PNG path")
    parser.add_argument("--title", type=str, default="Deep CFR Training Metrics", help="Chart title")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_metrics(args.metrics_file)
    out = Path(args.output)
    if out.parent and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)
    plot_metrics(data, args.title, args.output)
    print(f"Wrote chart to {args.output}")


if __name__ == "__main__":
    main()
