"""Render measured BS × ISL operating points and separate memory metrics.

Requires matplotlib. This script only reads benchmark JSON; it never runs or
loads a model. SVG/PNG and a compact CSV are suitable for the fork overview.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


BG, PANEL, INK, MUTED, GRID = "#10151e", "#151c27", "#eef2f7", "#aab6c8", "#2c3645"
COLORS = {"cpu": "#55c2b5", "mps": "#f0b357"}
MARKERS = {1: "o", 2: "s", 4: "^"}
BATCHES, LENGTHS = (1, 2, 4), (256, 512, 1024)


def style():
  plt.rcParams.update({
      "font.family": "DejaVu Sans", "font.size": 11,
      "figure.facecolor": BG, "axes.facecolor": PANEL,
      "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
      "ytick.color": MUTED, "axes.edgecolor": GRID, "grid.color": GRID,
      "grid.linewidth": .7, "axes.titleweight": "bold", "svg.fonttype": "none",
      "savefig.facecolor": BG})


def lookup(runs, device, batch, length):
  return next(x for x in runs if x["device"] == device and x["batch_size"] == batch
              and x["input_sequence_length"] == length)


def save(fig, destination, name):
  svg = destination / f"{name}.svg"
  fig.savefig(svg)
  svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
  fig.savefig(destination / f"{name}.png", dpi=160)
  plt.close(fig)


def tradeoff(runs, destination):
  fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.8))
  fig.subplots_adjust(left=.075, right=.985, bottom=.25, top=.74, wspace=.27)
  fig.text(.065, .95, "DetectorScope without CUDA", size=22, weight="bold")
  fig.text(.065, .898, "Throughput–latency operating points  ·  All parameters FP32  ·  Original checkpoint",
           size=11.5, color=MUTED)
  handles = [Line2D([0], [0], color=COLORS[d], lw=2, marker="o", label=f"Fork {d.upper()}")
             for d in ("cpu", "mps")]
  fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.061, .864),
             ncol=2, frameon=False, labelcolor=INK)
  largest_tps = max(x["metrics"]["input_token_throughput"] for x in runs if x["status"] == "complete")
  for ax, length in zip(axes, LENGTHS):
    ax.set_title(f"ISL = {length:,} tokens / chunk", loc="left", pad=17, size=13)
    for device in ("cpu", "mps"):
      rows = [lookup(runs, device, batch, length) for batch in BATCHES]
      complete = [x for x in rows if x["status"] == "complete"]
      ax.plot([x["metrics"]["e2e_latency_p50_seconds"] for x in complete],
              [x["metrics"]["input_token_throughput"] for x in complete],
              color=COLORS[device], alpha=.7, lw=1.5, linestyle="--", zorder=2)
      for row in complete:
        m, batch = row["metrics"], row["batch_size"]
        x, y = m["e2e_latency_p50_seconds"], m["input_token_throughput"]
        ax.errorbar(x, y,
            xerr=[[x - m["e2e_latency_min_seconds"]], [m["e2e_latency_max_seconds"] - x]],
            fmt="none", ecolor=COLORS[device], alpha=.45, capsize=3, lw=1, zorder=2)
        ax.scatter(x, y, s=78, marker=MARKERS[batch], color=COLORS[device],
                   edgecolors=BG, linewidths=1.2, zorder=3)
        dy = 12 if device == "cpu" or y < largest_tps * .15 else -18
        ax.annotate(f"BS={batch}", (x, y), xytext=(4, dy), textcoords="offset points",
                    color=COLORS[device], size=9.5, weight="bold")
      if any(x["status"] == "oom" for x in rows):
        failed = ", ".join(f"BS={x['batch_size']}" for x in rows if x["status"] == "oom")
        ax.text(.97, .96, f"{device.upper()} {failed}: OOM\n2 GiB MPS allocator cap",
                ha="right", va="top", transform=ax.transAxes, size=9, color="#ffaaa0",
                bbox={"facecolor": "#352329", "edgecolor": "#6c3a41", "boxstyle": "round,pad=.5"})
    ax.grid(True, alpha=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("P50 E2E request latency (s)", labelpad=12)
    ax.set_ylim(0, largest_tps * 1.2)
    ax.margins(x=.16)
    ax.spines[["top", "right"]].set_visible(False)
  axes[0].set_ylabel("Input token throughput (tokens/s)", labelpad=12)
  fig.text(.065, .105, "BS = independent chunks in one forward; ISL = tokens per chunk. OSL = 0: no generated token stream.",
           color=MUTED, size=10)
  fig.text(.065, .065, "M4 Pro · 24 GB · CPU threads 8 · one CPU prefetch block · 6 timed calls per cell after warmup · 2026-10-29 KST",
           color=MUTED, size=9.5)
  fig.text(.065, .025, "Lower latency / higher throughput is better. Whiskers: observed min–max, not confidence intervals. Lines connect measured BS settings.",
           color=MUTED, size=9.5)
  save(fig, destination, "mac-throughput-latency")


def memory(runs, destination):
  fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.7))
  fig.subplots_adjust(left=.065, right=.89, bottom=.25, top=.75, wspace=.3)
  fig.text(.065, .95, "Batch size × input length: memory", size=22, weight="bold")
  fig.text(.065, .895, "Process RSS and MPS driver allocation are separate measurements on unified memory; do not add them.",
           size=11, color=MUTED)
  specs = [("cpu", "rss", "CPU · peak process RSS"),
           ("mps", "rss", "MPS · peak process RSS"),
           ("mps", "driver", "MPS · observed driver maximum")]
  all_values = []
  matrices = []
  for device, kind, title in specs:
    matrix = np.full((3, 3), np.nan)
    for i, batch in enumerate(BATCHES):
      for j, length in enumerate(LENGTHS):
        row = lookup(runs, device, batch, length)
        if row["status"] == "complete":
          matrix[i, j] = (row["peak_process_rss_gib"] if kind == "rss" else
                          row["max_mps_driver_bytes_at_block_completion"] / 2**30)
          all_values.append(matrix[i, j])
    matrices.append(matrix)
  cmap = plt.get_cmap("viridis").copy()
  cmap.set_bad("#272d38")
  low, high = min(all_values), max(all_values)
  for ax, (device, kind, title), matrix in zip(axes, specs, matrices):
    image = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect="equal")
    ax.set_title(title, loc="left", pad=17, size=12)
    for i, batch in enumerate(BATCHES):
      for j, length in enumerate(LENGTHS):
        row = lookup(runs, device, batch, length)
        if row["status"] != "complete":
          ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                 hatch="///", edgecolor="#8b555a", linewidth=0))
          label, color = "OOM\n2 GiB cap", "#ffb5aa"
        else:
          value = matrix[i, j]
          label = f"{value:.2f}"
          color = INK if (value - low) / max(high - low, 1e-12) < .62 else "#102018"
        ax.text(j, i, label, ha="center", va="center", color=color, weight="bold",
                size=10 if row["status"] != "complete" else 13)
    ax.set_xticks(range(3), [f"{x:,}" for x in LENGTHS])
    ax.set_yticks(range(3), [str(x) for x in BATCHES])
    ax.set_xlabel("ISL (tokens / chunk)", labelpad=10)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color=BG, linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
  axes[0].set_ylabel("Batch size (chunks / forward)", labelpad=12)
  colorbar = fig.colorbar(image, ax=axes, fraction=.025, pad=.03)
  colorbar.set_label("GiB", labelpad=10)
  fig.text(.065, .13, "Each cell uses a fresh process. Peak RSS includes initialization and timed calls; driver memory is sampled at block completion.",
           color=MUTED, size=9.6)
  fig.text(.065, .075, "Failed configurations are marked OOM, not filled with zero or interpolated. All parameters FP32 · M4 Pro / 24 GB · 2026-10-29 KST",
           color=MUTED, size=9.5)
  save(fig, destination, "mac-memory-grid")


def export_csv(runs, destination):
  fields = ["backend", "batch_size", "input_sequence_length", "output_sequence_length", "precision",
            "status", "timed_calls", "e2e_latency_p50_s", "input_token_throughput_tok_s",
            "backbone_prefill_token_throughput_tok_s", "peak_process_rss_gib",
            "observed_mps_driver_max_gib", "interactivity_tok_s_user"]
  with (destination / "mac-benchmark-grid.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(runs, key=lambda x: (x["device"], x["input_sequence_length"], x["batch_size"])):
      complete = row["status"] == "complete"
      m = row.get("metrics", {})
      writer.writerow({
          "backend": row["device"], "batch_size": row["batch_size"],
          "input_sequence_length": row["input_sequence_length"], "output_sequence_length": 0,
          "precision": "FP32", "status": row["status"], "timed_calls": len(row["requests"]),
          "e2e_latency_p50_s": m.get("e2e_latency_p50_seconds", ""),
          "input_token_throughput_tok_s": m.get("input_token_throughput", ""),
          "backbone_prefill_token_throughput_tok_s": m.get("backbone_prefill_token_throughput", ""),
          "peak_process_rss_gib": row["peak_process_rss_gib"] if complete else "",
          "observed_mps_driver_max_gib": (row["max_mps_driver_bytes_at_block_completion"] / 2**30
                                          if complete and row["device"] == "mps" else ""),
          "interactivity_tok_s_user": "N/A (no decode)",
      })


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, default=Path("outputs/mac-grid-20261029/summary.json"))
  parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
  args = parser.parse_args()
  report = json.loads(args.input.read_text())
  runs = report["runs"]
  if report["status"] != "complete" or len(runs) != 18:
    raise ValueError("A complete joint sweep is required; missing cells must not be invented.")
  if any(row["status"] not in ("complete", "oom") for row in runs):
    raise ValueError("Errors or unfinished cells must not be labeled as OOM.")
  args.assets.mkdir(parents=True, exist_ok=True)
  style()
  export_csv(runs, args.assets)
  tradeoff(runs, args.assets)
  memory(runs, args.assets)
  print(f"Wrote two SVG/PNG figures and one measured-data CSV to {args.assets}")


if __name__ == "__main__":
  main()
