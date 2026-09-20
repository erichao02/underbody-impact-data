#!/usr/bin/env python3
"""Load automotive-impact cases from extracted files or ZIP shards."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch


GEOMETRY_ALIASES = {
    "floorfrontdriver": "floorfrontdriver",
    "driver": "floorfrontdriver",
    "floorfrontr": "floorfrontR",
    "floorfrontR": "floorfrontR",
    "trunk": "trunkfloor",
    "trunkfloor": "trunkfloor",
}


def canonical_geometry(name: str) -> str:
    if name in GEOMETRY_ALIASES:
        return GEOMETRY_ALIASES[name]
    lowered = name.lower()
    if lowered in GEOMETRY_ALIASES:
        return GEOMETRY_ALIASES[lowered]
    choices = ", ".join(sorted(set(GEOMETRY_ALIASES.values())))
    raise ValueError(f"unknown geometry {name!r}; choose one of {choices}")


def canonical_case_id(value: str | int) -> str:
    if isinstance(value, int):
        number = value
    else:
        text = str(value).strip()
        if text.lower().endswith(".pt"):
            text = text[:-3]
        if text.lower().startswith("case"):
            text = text[4:]
        number = int(text)
    if not 1 <= number <= 500:
        raise ValueError(f"case number must be in [1,500], got {number}")
    return f"case{number:03d}"


def shard_name(case_id: str) -> str:
    number = int(case_id[4:])
    start = ((number - 1) // 100) * 100 + 1
    end = start + 99
    return f"cases_{start:03d}_{end:03d}.zip"


def load_case_bytes(dataset_root: Path | str, geometry: str, case_id: str | int) -> bytes:
    root = Path(dataset_root)
    geometry = canonical_geometry(geometry)
    case_id = canonical_case_id(case_id)

    candidates = (
        root / "data" / geometry / "cases" / f"{case_id}.pt",
        root / "data" / geometry / f"{case_id}.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_bytes()

    archive = root / "data" / geometry / shard_name(case_id)
    if not archive.is_file():
        raise FileNotFoundError(
            f"case not extracted and shard is missing: {archive}"
        )
    member = f"cases/{case_id}.pt"
    with zipfile.ZipFile(archive) as handle:
        try:
            return handle.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{member} is missing from {archive}") from exc


def load_case(dataset_root: Path | str, geometry: str, case_id: str | int) -> dict:
    payload = load_case_bytes(dataset_root, geometry, case_id)
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def load_mesh(dataset_root: Path | str, geometry: str) -> dict[str, np.ndarray]:
    root = Path(dataset_root)
    geometry = canonical_geometry(geometry)
    path = root / "meshes" / f"{geometry}_mesh.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_split(dataset_root: Path | str) -> dict[str, list[str]]:
    path = Path(dataset_root) / "metadata" / "split_400_50_50_seed12345.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    geometry = canonical_geometry(args.geometry)
    case_id = canonical_case_id(args.case)
    data = load_case(args.dataset_root, geometry, case_id)
    mesh = load_mesh(args.dataset_root, geometry)

    summary = {
        "geometry": geometry,
        "case": data["case"],
        "disp_shape": list(data["disp"].shape),
        "effective_stress_shape": list(data["effective_stress"].shape),
        "time": data["time"].tolist(),
        "impact_xyz": data["impact_xyz"].tolist(),
        "velocity_xyz": data["velocity_xyz"].tolist(),
        "mass_ratio": float(data["mass_ratio"]),
        "material_young_mpa": float(data["material_young_mpa"]),
        "material_poisson": float(data["material_poisson"]),
        "mesh_node_count": int(mesh["node_pos"].shape[0]),
        "mesh_element_count": int(mesh["element_node_index"].shape[0]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
