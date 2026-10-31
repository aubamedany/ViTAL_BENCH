#!/usr/bin/env python3
"""Render Fig. 3 in the original ViTALBench figure style."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (ROOT / "paper/Figures/Statistic.pdf", ROOT / "paper/_aaai/Figures/Statistic.pdf")


def load_data() -> tuple[list[dict], list[dict], list[dict]]:
    detection = json.loads((ROOT / "detection.json").read_text())
    law = json.loads((ROOT / "law.json").read_text())
    law_categories = json.loads((ROOT / "law_categories.json").read_text())
    return detection, law, law_categories


def draw(detection: list[dict], law: list[dict], law_categories: list[dict]) -> plt.Figure:
    detection_counts = Counter(item["type_image"] for item in detection)
    law_counts = Counter(item["type_image"] for item in law)
    categories = sorted(detection_counts | law_counts)
    detection_values = [detection_counts[category] for category in categories]
    law_values = [law_counts[category] for category in categories]
    labels = [category.replace("_", " ").title() for category in categories]

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 14, "axes.labelsize": 10})
    fig, (pie_ax, bar_ax) = plt.subplots(1, 2, figsize=(15, 5.1), gridspec_kw={"width_ratios": [0.85, 1.55]})

    category_names = [
        "Violation identification",
        "Regulation Application",
        "Multi-step legal Reasoning",
    ]
    category_values = [sum(item["category"] == name for item in law_categories) for name in category_names]
    pie_values = [len(detection), *category_values]
    pie_labels = ["Detection", *(f"{name} (Law)" for name in category_names)]
    pie_colors = ["#7da8c7", "#80c8bd", "#fa7d73", "#f9b35d"]
    total = sum(pie_values)
    pie_ax.pie(
        pie_values,
        colors=pie_colors,
        startangle=90,
        counterclock=False,
        autopct=lambda value: f"{value:.1f}%\n({round(value * total / 100):,})",
        textprops={"fontsize": 11, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    pie_ax.set_title("Task / category split", fontweight="bold", pad=14)
    pie_ax.legend(pie_labels, loc="lower left", bbox_to_anchor=(-0.02, -0.02), frameon=False, fontsize=9)

    x = list(range(len(categories)))
    width = 0.28
    bar_ax.bar(
        [index - width / 2 for index in x],
        detection_values,
        width,
        label="Detection",
        color="#9ec3da",
        edgecolor="#555555",
        hatch="///",
        linewidth=0.7,
    )
    bar_ax.bar(
        [index + width / 2 for index in x],
        law_values,
        width,
        label="Law",
        color="#bfc3c7",
        edgecolor="#555555",
        linewidth=0.7,
    )
    bar_ax.set_title(
        f"Image category distribution - Detection (n={len(detection):,}) vs Law (n={len(law):,})",
        fontweight="bold",
        pad=14,
    )
    bar_ax.set_ylabel("count")
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9)
    bar_ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    bar_ax.set_axisbelow(True)
    bar_ax.legend(loc="upper right", frameon=False, fontsize=9)
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)
    bar_ax.set_ylim(0, max(law_values) * 1.18)
    for index, value in enumerate(detection_values):
        bar_ax.text(index - width / 2, value + 8, f"{value}", ha="center", va="bottom", fontsize=8)
    for index, value in enumerate(law_values):
        bar_ax.text(index + width / 2, value + 8, f"{value}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    figure = draw(*load_data())
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, format="pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
