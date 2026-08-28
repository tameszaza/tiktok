"""Plots for the technique-vs-shape sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .technique_shape import TECHNIQUES, TechniqueResult


def write_technique_shape_plot(path: Path, results: list[TechniqueResult]) -> None:
    shapes = []
    for result in results:
        if result.shape.label not in shapes:
            shapes.append(result.shape.label)
    techniques = [name for name, _ in TECHNIQUES]
    matrix = np.full((len(shapes), len(techniques)), np.nan)
    for result in results:
        matrix[shapes.index(result.shape.label), techniques.index(result.technique)] = result.speedup
    fig, (heat_ax, bar_ax) = plt.subplots(
        1, 2,
        figsize=(max(24, len(techniques) * 0.9), max(14, len(shapes) * 0.35)),
        gridspec_kw={"width_ratios": (1.8, 1)},
        constrained_layout=True,
    )
    image = heat_ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=min(0.8, np.nanmin(matrix)), vmax=max(1.0, np.nanmax(matrix)))
    heat_ax.set_xticks(np.arange(len(techniques)), techniques, rotation=90)
    heat_ax.set_yticks(np.arange(len(shapes)), shapes)
    heat_ax.set_xlabel("optimization technique")
    heat_ax.set_ylabel("model shape")
    heat_ax.set_title("Speedup by technique and model shape")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value):
                heat_ax.text(col, row, f"{value:.2f}×", ha="center", va="center", fontsize=8, color="black" if value < 1.45 else "white")
    fig.colorbar(image, ax=heat_ax, label="speedup (baseline / optimized)")
    means = [float(np.nanmean(matrix[:, index])) for index in range(len(techniques))]
    bars = bar_ax.bar(techniques, means, color="#238b45")
    bar_ax.axhline(1.0, color="#555555", linewidth=1)
    bar_ax.set_ylabel(f"mean speedup across {len(shapes)} shapes")
    bar_ax.set_title("Technique average")
    bar_ax.tick_params(axis="x", rotation=90)
    for bar, value in zip(bars, means):
        bar_ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}×", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Technique × shape experiment ({len(results)} measurements)", fontsize=15)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
