#!/usr/bin/env python3
"""Plot defense-state and gradient-probe records as publication-style 2D scatters."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


OPINION_ORDER = ["opinion_normal", "opinion_misaligned1", "opinion_misaligned2"]
COLOR_MAP = {
    "opinion_normal": "#1f77b4",       # blue
    "opinion_misaligned1": "#ff7f0e",   # orange
    "opinion_misaligned2": "#2ca02c",   # green
}


def _iter_jsonl_points(paths: Iterable[Path], x_key: str, y_key: str) -> Dict[str, List[Tuple[float, float]]]:
    points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for path in sorted(paths):
        parent = path.parent.name
        if "_opinion_" not in parent:
            continue
        opinion = f"opinion_{parent.rsplit('_opinion_', 1)[1]}"
        if opinion not in OPINION_ORDER:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                points[opinion].append((float(record[x_key]), float(record[y_key])))
    return points


def _collect_paths(root: Path, method_dir: str) -> List[Path]:
    return sorted((root / method_dir).glob("*.jsonl"))


def _discover_prefix_groups(root: Path) -> Dict[str, Dict[str, List[Path]]]:
    groups: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if "_opinion_" not in folder.name:
            continue
        prefix, opinion = folder.name.rsplit("_opinion_", 1)
        opinion = f"opinion_{opinion}"
        if opinion not in OPINION_ORDER:
            continue
        groups[prefix][opinion].extend(sorted(folder.glob("*.jsonl")))
    return groups


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.22)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.10)
    ax.tick_params(axis="both", labelsize=10, length=4, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _scatter_by_opinion(ax: plt.Axes, points: Dict[str, List[Tuple[float, float]]], *, xlabel: str, ylabel: str, title: str) -> None:
    _style_axes(ax, title, xlabel, ylabel)

    for opinion in OPINION_ORDER:
        coords = points.get(opinion, [])
        if not coords:
            continue
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        ax.scatter(
            xs,
            ys,
            s=8,
            alpha=0.28,
            linewidths=0,
            c=COLOR_MAP[opinion],
            label=opinion.replace("opinion_", "").replace("misaligned", "misaligned "),
            rasterized=True,
        )

    if any(points.get(opinion) for opinion in OPINION_ORDER):
        ax.legend(
            frameon=False,
            fontsize=10,
            loc="best",
            handletextpad=0.4,
            borderaxespad=0.2,
            scatterpoints=1,
            markerscale=2.0,
        )


def _apply_margins(ax: plt.Axes, xs: List[float], ys: List[float], pad_ratio: float = 0.06) -> None:
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max((x_max - x_min) * pad_ratio, 1e-12)
    y_pad = max((y_max - y_min) * pad_ratio, 1e-12)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _configure_scientific_axis(ax: plt.Axes, axis: str) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot defense records for defense_state and gradient_probe.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("deepspeed-chat/defense_records/Qwen3-8B"),
        help="Root defense_records directory for one model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("deepspeed-chat/defense_records/Qwen3-8B/plots"),
        help="Where to save the generated figures.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": args.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = _discover_prefix_groups(args.root)
    if not groups:
        raise FileNotFoundError(f"No *_opinion_* folders found under {args.root}")

    for prefix in sorted(groups):
        opinion_dirs = groups[prefix]
        paths: List[Path] = []
        for opinion in OPINION_ORDER:
            paths.extend(opinion_dirs.get(opinion, []))
        if not paths:
            continue

        if prefix.startswith("gradient_probe"):
            x_key, y_key = "probe_cos_inj", "probe_proj_inj"
            xlabel = r"$\mathrm{probe\_cos\_inj}$"
            ylabel = r"$\mathrm{probe\_proj\_inj}$"
        elif prefix.startswith("defense_state"):
            x_key, y_key = "delta_p", "p_current"
            xlabel = r"$\mathrm{\Delta p}$"
            ylabel = r"$p_{\mathrm{current}}$"
        else:
            continue

        points = _iter_jsonl_points(paths, x_key, y_key)
        fig, ax = plt.subplots(figsize=(6.8, 5.6))
        _scatter_by_opinion(
            ax,
            points,
            xlabel=xlabel,
            ylabel=ylabel,
            title=prefix.replace("_", " "),
        )
        _apply_margins(
            ax,
            [x for pts in points.values() for x, _ in pts],
            [y for pts in points.values() for _, y in pts],
            pad_ratio=0.07,
        )
        if prefix.startswith("gradient_probe"):
            _configure_scientific_axis(ax, "y")

        out_path = args.output_dir / f"{prefix}_scatter.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
