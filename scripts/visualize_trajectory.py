#!/usr/bin/env python3
"""Render nodal displacement magnitude and shell von Mises stress by state."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from load_case import canonical_geometry, load_case, load_mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--state", type=int, default=0, choices=range(17))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    geometry = canonical_geometry(args.geometry)
    data = load_case(args.dataset_root, geometry, args.case)
    mesh = load_mesh(args.dataset_root, geometry)
    state = args.state

    pos = mesh["node_pos"]
    displacement = data["disp"][:, state].numpy()
    deformed = pos + displacement
    magnitude = torch.linalg.vector_norm(data["disp"][:, state], dim=-1).numpy()
    connectivity = mesh["element_node_index"]
    centers = deformed[connectivity].mean(axis=1)
    stress = data["effective_stress"][:, state].numpy()

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    first = axes[0].scatter(
        deformed[:, 0], deformed[:, 1], c=magnitude, s=1.2, cmap="viridis"
    )
    axes[0].set_title(f"Nodal displacement magnitude, state {state}")
    figure.colorbar(first, ax=axes[0])
    second = axes[1].scatter(
        centers[:, 0], centers[:, 1], c=stress, s=1.2, cmap="turbo"
    )
    axes[1].set_title(f"Max-IP von Mises stress, state {state}")
    figure.colorbar(second, ax=axes[1])
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x (unit pending confirmation)")
        axis.set_ylabel("y (unit pending confirmation)")
    figure.suptitle(f"{geometry} / {data['case']} / t={float(data['time'][state]):.6g}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=200)
        print(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()
