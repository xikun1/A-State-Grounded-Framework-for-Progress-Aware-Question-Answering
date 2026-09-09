"""Plot stale-state and distractor-evidence robustness."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _plot_common import configure, read_csv, save


def main() -> None:
    configure()
    stale = read_csv("stale_state_robustness.csv")
    distractor = read_csv("distractor_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(
        [int(row["state_staleness"]) for row in stale],
        [float(row["action_validity_rate"]) for row in stale],
        marker="o",
        linewidth=2.2,
        color="#C53030",
    )
    axes[0].set_xticks([0, 1, 2, 3])
    axes[0].set_xlabel("State staleness Δ")
    axes[0].set_ylabel("Action validity rate (%)")
    axes[0].set_title("(a) Stale runtime state")
    methods = ["Standard RAG", "State-conditioned RAG", "Full Method – Progress/Action"]
    colors = ["#718096", "#3182CE", "#C53030"]
    for method, color in zip(methods, colors):
        rows = [row for row in distractor if row["method"] == method]
        axes[1].plot(
            [int(row["distractor_level"]) for row in rows],
            [float(row["next_action_accuracy"]) for row in rows],
            marker="o",
            linewidth=2,
            color=color,
            label=method,
        )
    axes[1].set_xticks([0, 50, 100, 200], ["0%", "50%", "100%", "200%"])
    axes[1].set_xlabel("Distractor contamination")
    axes[1].set_ylabel("Next-action accuracy (%)")
    axes[1].set_title("(b) Metadata-admissible distractors")
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, "fig5")


if __name__ == "__main__":
    main()
