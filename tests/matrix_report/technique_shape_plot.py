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
    result_by_cell = {}
    for result in results:
        matrix[shapes.index(result.shape.label), techniques.index(result.technique)] = result.speedup
        result_by_cell[(result.shape.label, result.technique)] = result
    fig, (heat_ax, bar_ax) = plt.subplots(
        1, 2,
        figsize=(max(24, len(techniques) * 0.9), max(14, len(shapes) * 0.35)),
        gridspec_kw={"width_ratios": (1.8, 1)},
        constrained_layout=True,
    )
    finite_values = matrix[np.isfinite(matrix)]
    vmin = min(0.8, float(finite_values.min())) if finite_values.size else 0.8
    vmax = max(1.0, float(finite_values.max())) if finite_values.size else 1.0
    image = heat_ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=vmin, vmax=vmax)
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
            else:
                result = result_by_cell[(shapes[row], techniques[col])]
                if np.isfinite(result.optimized_ms):
                    text = f"B:OOM\nO:{result.optimized_ms:.1f}ms"
                else:
                    text = "OOM"
                heat_ax.text(col, row, text, ha="center", va="center", fontsize=6, color="#555555")
    fig.colorbar(image, ax=heat_ax, label="speedup (baseline / optimized)")
    means = [
        float(values.mean()) if (values := matrix[:, index][np.isfinite(matrix[:, index])]).size else float("nan")
        for index in range(len(techniques))
    ]
    bars = bar_ax.bar(
        techniques,
        [value if np.isfinite(value) else 0.0 for value in means],
        color="#238b45",
    )
    bar_ax.axhline(1.0, color="#555555", linewidth=1)
    bar_ax.set_ylabel(f"mean speedup across {len(shapes)} shapes")
    bar_ax.set_title("Technique average")
    bar_ax.tick_params(axis="x", rotation=90)
    for bar, value in zip(bars, means):
        label = f"{value:.2f}×" if np.isfinite(value) else "—"
        height = value if np.isfinite(value) else 0.0
        bar_ax.text(bar.get_x() + bar.get_width() / 2, height, label, ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Technique × shape experiment ({len(results)} measurements)", fontsize=15)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
