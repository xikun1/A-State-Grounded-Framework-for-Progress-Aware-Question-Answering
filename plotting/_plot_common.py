"""Shared paths and style for manuscript figure generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "reported_metrics"
FIGURES = ROOT / "results" / "figures"


def read_csv(name: str) -> list[dict[str, str]]:
    """Read one reported aggregate-metrics table."""

    with (METRICS / name).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def configure() -> None:
    """Apply a compact, publication-friendly Matplotlib style."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def save(fig: Any, stem: str) -> None:
    """Save PNG and vector PDF outputs under results/figures."""

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
