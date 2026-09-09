"""Plot verifier detection recall and residual violations by error type."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _plot_common import configure, read_csv, save


def main() -> None:
    configure()
    rows = [row for row in read_csv("verifier_results.csv") if row["source"] == "Table 10"]
    labels = [
        "Schema violation",
        "Scene mismatch",
        "Evidence mismatch",
        "Progress mismatch",
        "Order violation",
        "Illegal action",
        "Duplicate step",
    ]
    lookup = {(row["violation_type"], row["metric"]): float(row["value"]) for row in rows}
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    ax.bar(x - width / 2, [lookup[(label, "detection_recall")] for label in labels], width, label="Detection recall", color="#2B6CB0")
    ax.bar(x + width / 2, [lookup[(label, "residual_violation_rate")] for label in labels], width, label="Residual violation", color="#DD6B20")
    ax.set_xticks(x, [label.replace(" ", "\n") for label in labels])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig3")


if __name__ == "__main__":
    main()
