"""Plot action validity across task-progress stages."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _plot_common import configure, read_csv, save


def main() -> None:
    configure()
    rows = [row for row in read_csv("progress_action.csv") if row["scope"] == "progress bins"]
    stages = ["0-25%", "25-50%", "50-75%", "75-100%"]
    methods = ["Standard RAG", "ProgPrompt", "ExRAP", "Full Method – Progress/Action"]
    colors = ["#718096", "#3182CE", "#38A169", "#C53030"]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for method, color in zip(methods, colors):
        lookup = {row["progress_stage"]: float(row["value"]) for row in rows if row["method"] == method}
        ax.plot(stages, [lookup[stage] for stage in stages], marker="o", linewidth=2, label=method, color=color)
    ax.set_xlabel("Task progress stage")
    ax.set_ylabel("Action validity rate (%)")
    ax.set_ylim(50, 100)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "fig2")


if __name__ == "__main__":
    main()
