#!/usr/bin/env python3
"""Derive displacement-peak snapshots from the released 17-state trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch

from load_case import canonical_case_id, canonical_geometry, load_case, load_split


def select_peak(data: dict) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    displacement = data["disp"].to(torch.float32)
    stress = data["effective_stress"].to(torch.float32)
    valid = data["valid_node_mask"].to(torch.bool)
    if displacement.ndim != 3 or displacement.shape[1:] != (17, 3):
        raise ValueError(f"invalid displacement shape: {tuple(displacement.shape)}")
    if stress.ndim != 2 or stress.shape[1] != 17:
        raise ValueError(f"invalid effective-stress shape: {tuple(stress.shape)}")
    magnitude = torch.linalg.vector_norm(displacement, dim=-1)
    magnitude = magnitude.masked_fill(~valid[:, None], float("-inf"))
    flat_index = int(torch.argmax(magnitude).item())
    time_index = flat_index % displacement.shape[1]
    node_index = flat_index // displacement.shape[1]
    return (
        time_index,
        node_index,
        displacement[:, time_index, :].contiguous(),
        stress[:, time_index].contiguous(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Optional case identifiers; default is all 500 cases.",
    )
    args = parser.parse_args()

    geometry = canonical_geometry(args.geometry)
    case_ids = (
        [canonical_case_id(case) for case in args.cases]
        if args.cases
        else [f"case{index:03d}" for index in range(1, 501)]
    )
    output_cases = args.output_root / "cases"
    output_cases.mkdir(parents=True, exist_ok=True)

    rows = []
    for offset, case_id in enumerate(case_ids, start=1):
        data = load_case(args.dataset_root, geometry, case_id)
        time_index, node_index, displacement, stress = select_peak(data)
        peak = {
            "case": case_id,
            "selected_time_index": torch.tensor(time_index, dtype=torch.int64),
            "selected_time": data["time"][time_index].to(torch.float32),
            "disp_peak_value": torch.linalg.vector_norm(
                displacement[node_index]
            ).to(torch.float32),
            "disp_peak_node_index": torch.tensor(node_index, dtype=torch.int64),
            "disp": displacement,
            "element_results": {"effective_stress": stress},
        }
        for key in (
            "impact_xyz",
            "velocity_xyz",
            "mass_ratio",
            "material_young_mpa",
            "material_poisson",
            "boundary_mask",
            "valid_node_mask",
        ):
            peak[key] = data[key]
        torch.save(peak, output_cases / f"{case_id}.pt")
        rows.append(
            {
                "case": case_id,
                "selected_time_index": time_index,
                "selected_time": float(data["time"][time_index]),
                "disp_peak_node_index": node_index,
                "disp_peak_value": float(peak["disp_peak_value"]),
            }
        )
        if offset % 50 == 0 or offset == len(case_ids):
            print(f"[{geometry}] derived {offset}/{len(case_ids)}")

    with (args.output_root / "disp_peak_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split = load_split(args.dataset_root)
    (args.output_root / "split_400_50_50_seed12345.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )
    source_mesh = (
        args.dataset_root / "meshes" / f"{geometry}_mesh.npz"
    )
    shutil.copy2(source_mesh, args.output_root / "mesh.npz")


if __name__ == "__main__":
    main()

