#!/usr/bin/env python3
"""Compute paper-protocol peak-field normalization from training cases only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build_peak_targets import select_peak
from load_case import canonical_geometry, load_case, load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    geometry = canonical_geometry(args.geometry)
    train_cases = load_split(args.dataset_root)["train"]
    target_sum_sq = torch.zeros(4, dtype=torch.float64)
    target_count = torch.zeros(4, dtype=torch.int64)
    condition_rows = []

    for offset, case_id in enumerate(train_cases, start=1):
        data = load_case(args.dataset_root, geometry, case_id)
        _, _, displacement, stress = select_peak(data)
        for component in range(3):
            values = displacement[:, component].to(torch.float64)
            target_sum_sq[component] += torch.sum(values.square())
            target_count[component] += values.numel()
        stress64 = stress.to(torch.float64)
        target_sum_sq[3] += torch.sum(stress64.square())
        target_count[3] += stress64.numel()
        condition_rows.append(
            torch.cat(
                (
                    data["velocity_xyz"].to(torch.float64).reshape(3),
                    data["mass_ratio"].to(torch.float64).reshape(1),
                    data["material_young_mpa"].to(torch.float64).reshape(1),
                    data["material_poisson"].to(torch.float64).reshape(1),
                )
            )
        )
        if offset % 50 == 0:
            print(f"[{geometry}] normalization {offset}/{len(train_cases)}")

    scale = torch.sqrt(target_sum_sq / target_count)
    conditions = torch.stack(condition_rows)
    payload = {
        "geometry": geometry,
        "split": "train",
        "train_cases": len(train_cases),
        "target_scale_definition": (
            "training-split RMS of peak-state x/y/z displacement and "
            "same-state max-IP von Mises effective stress"
        ),
        "target_feature_names": ["disp_x", "disp_y", "disp_z", "effective_stress"],
        "target_scale": scale.tolist(),
        "condition_feature_names": [
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "mass_ratio",
            "material_young_mpa",
            "material_poisson",
        ],
        "condition_mean": conditions.mean(dim=0).tolist(),
        "condition_std_sample": conditions.std(dim=0, unbiased=True).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
