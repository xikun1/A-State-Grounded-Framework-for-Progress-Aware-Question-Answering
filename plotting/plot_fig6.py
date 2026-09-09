"""Plot track-specific quality versus mean request latency."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _plot_common import configure, read_csv, save


METHODS = ["LLM-only", "Standard RAG", "State-only", "State-conditioned RAG", "Full Method"]
COLORS = {
    "LLM-only": "#718096",
    "Standard RAG": "#805AD5",
    "State-only": "#2B6CB0",
    "State-conditioned RAG": "#38A169",
    "Full Method": "#C53030",
}


def main() -> None:
    configure()
    efficiency = read_csv("efficiency.csv")
    progress = read_csv("progress_action.csv")
    ue5 = read_csv("ue5_replay_summary.csv")
    latency = {(row["track"], row["method"]): float(row["mean_latency_s"]) for row in efficiency}
    eai_quality = {
        ("Full Method" if row["method"].startswith("Full Method") else row["method"]): float(row["value"])
        for row in progress
        if row["source"] == "Table 6" and row["metric"] == "next_action_accuracy"
    }
    ue5_quality = {
        ("Full Method" if row["method"].startswith("Full Method") else row["method"]): float(row["value"])
        for row in ue5
        if row["source"] == "Table 13" and row["metric"] == "support_rate"
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    panels = [
        (axes[0], "EAI–VirtualHome", eai_quality, "Next-action accuracy (%)", "(a) EAI–VirtualHome"),
        (axes[1], "UE5 Offline Replay", ue5_quality, "Support rate (%)", "(b) UE5 offline replay"),
    ]
    for ax, track, quality, ylabel, title in panels:
        for method in METHODS:
            x, y = latency[(track, method)], quality[method]
            ax.scatter(x, y, s=52, color=COLORS[method], label=method, zorder=3)
            ax.annotate(method, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel("Mean latency (s/request)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig6")


if __name__ == "__main__":
    main()
