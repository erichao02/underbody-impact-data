#!/usr/bin/env python3
"""Validate the complete automotive-impact release package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from load_case import (
    canonical_case_id,
    load_case,
    load_case_bytes,
    load_mesh,
    load_split,
    shard_name,
)


SPECS = {
    "floorfrontdriver": {"nodes": 7408, "edges": 29572, "elements": 7374},
    "floorfrontR": {"nodes": 12011, "edges": 48138, "elements": 12055},
    "trunkfloor": {"nodes": 14440, "edges": 58074, "elements": 14589},
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"ERROR: {message}", file=sys.stderr)


def validate_split(root: Path, errors: list[str]) -> dict:
    split = load_split(root)
    expected_sizes = {"train": 400, "val": 50, "test": 50}
    for name, expected in expected_sizes.items():
        values = split.get(name)
        if not isinstance(values, list) or len(values) != expected:
            fail(errors, f"split {name!r}: expected {expected} entries")
            continue
        canonical = [canonical_case_id(value) for value in values]
        if canonical != values:
            fail(errors, f"split {name!r} contains non-canonical case IDs")
        if len(set(values)) != len(values):
            fail(errors, f"split {name!r} contains duplicates")
    union = set().union(*(set(split.get(key, [])) for key in expected_sizes))
    if union != {f"case{index:03d}" for index in range(1, 501)}:
        fail(errors, "split union is not exactly case001..case500")
    names = list(expected_sizes)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if set(split.get(left, [])) & set(split.get(right, [])):
                fail(errors, f"split overlap: {left} and {right}")
    return split


def validate_mesh(root: Path, geometry: str, spec: dict, errors: list[str]) -> dict:
    mesh = load_mesh(root, geometry)
    expected = {
        "node_pos": (spec["nodes"], 3),
        "edge_index": (2, spec["edges"]),
        "element_node_index": (spec["elements"], 4),
        "element_node_count": (spec["elements"],),
        "boundary_mask": (spec["nodes"],),
    }
    for key, shape in expected.items():
        if key not in mesh:
            fail(errors, f"{geometry} mesh missing {key}")
        elif mesh[key].shape != shape:
            fail(errors, f"{geometry} mesh {key}: {mesh[key].shape} != {shape}")
    if "node_pos" in mesh and not np.isfinite(mesh["node_pos"]).all():
        fail(errors, f"{geometry} mesh node_pos is non-finite")
    if "edge_index" in mesh:
        edges = mesh["edge_index"]
        if edges.min() < 0 or edges.max() >= spec["nodes"]:
            fail(errors, f"{geometry} edge_index is out of range")
    if "element_node_index" in mesh:
        connectivity = mesh["element_node_index"]
        if connectivity.min() < 0 or connectivity.max() >= spec["nodes"]:
            fail(errors, f"{geometry} element_node_index is out of range")
    if "element_node_count" in mesh and not np.isin(
        mesh["element_node_count"], [3, 4]
    ).all():
        fail(errors, f"{geometry} element_node_count contains values outside 3/4")
    return {
        "nodes": spec["nodes"],
        "edges": spec["edges"],
        "elements": spec["elements"],
    }


def validate_case(
    root: Path,
    geometry: str,
    case_id: str,
    spec: dict,
    expected_digest: str | None,
    errors: list[str],
) -> dict:
    payload = load_case_bytes(root, geometry, case_id)
    digest = sha256_bytes(payload)
    if expected_digest and digest != expected_digest:
        fail(errors, f"{geometry}/{case_id}: SHA-256 mismatch")
    data = torch.load(
        __import__("io").BytesIO(payload), map_location="cpu", weights_only=True
    )
    if data.get("case") != case_id:
        fail(errors, f"{geometry}/{case_id}: stored case identifier mismatch")
    expected_shapes = {
        "disp": (spec["nodes"], 17, 3),
        "effective_stress": (spec["elements"], 17),
        "time": (17,),
        "element_time": (17,),
        "valid_node_mask": (spec["nodes"],),
        "boundary_mask": (spec["nodes"],),
        "impact_xyz": (3,),
        "velocity_xyz": (3,),
    }
    for key, shape in expected_shapes.items():
        value = data.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            actual = None if value is None else getattr(value, "shape", type(value))
            fail(errors, f"{geometry}/{case_id} {key}: {actual} != {shape}")
    for key in ("disp", "effective_stress", "time", "element_time"):
        value = data.get(key)
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            fail(errors, f"{geometry}/{case_id}: {key} is non-finite")
    time_values = data.get("time")
    element_time = data.get("element_time")
    if torch.is_tensor(time_values) and time_values.numel() == 17:
        if not torch.all(time_values[1:] >= time_values[:-1]):
            fail(errors, f"{geometry}/{case_id}: time is not monotonic")
    if torch.is_tensor(element_time) and element_time.numel() == 17:
        if not torch.all(element_time[1:] >= element_time[:-1]):
            fail(errors, f"{geometry}/{case_id}: element_time is not monotonic")
    if (
        torch.is_tensor(time_values)
        and torch.is_tensor(element_time)
        and tuple(time_values.shape) == (17,)
        and tuple(element_time.shape) == (17,)
        and not torch.allclose(time_values, element_time, rtol=0.0, atol=1.0e-8)
    ):
        fail(errors, f"{geometry}/{case_id}: displacement/stress time arrays differ")
    return {
        "bytes": len(payload),
        "sha256": digest,
        "material": str(data.get("material_name", "")),
    }


def load_manifest(root: Path, errors: list[str]) -> dict[tuple[str, str], dict]:
    path = root / "manifest.csv"
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["geometry"], row["case"])
            if key in rows:
                fail(errors, f"duplicate manifest row: {key}")
            rows[key] = row
    if len(rows) != 1500:
        fail(errors, f"manifest has {len(rows)} rows, expected 1500")
    return rows


def validate_archives(root: Path, errors: list[str]) -> dict:
    archive_count = 0
    total_bytes = 0
    for geometry in SPECS:
        for start in range(1, 501, 100):
            end = start + 99
            path = root / "data" / geometry / f"cases_{start:03d}_{end:03d}.zip"
            if not path.is_file():
                fail(errors, f"missing archive: {path.relative_to(root)}")
                continue
            archive_count += 1
            total_bytes += path.stat().st_size
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                if bad_member:
                    fail(errors, f"CRC failure in {path.name}: {bad_member}")
                expected = {
                    f"cases/case{index:03d}.pt" for index in range(start, end + 1)
                }
                if set(archive.namelist()) != expected:
                    fail(errors, f"unexpected members in {path.relative_to(root)}")
    return {"archives": archive_count, "archive_bytes": total_bytes}


def validate_case_files(root: Path, manifest: dict, errors: list[str]) -> dict:
    """Check the complete expanded layout without requiring historical ZIPs."""
    expected = {
        (geometry, f"case{index:03d}")
        for geometry in SPECS for index in range(1, 501)
    }
    if set(manifest) != expected:
        fail(errors, "manifest keys do not match the 1,500 expected cases")
    expected_paths = set()
    total_bytes = 0
    for geometry, case_id in sorted(expected):
        relative = f"data/{geometry}/cases/{case_id}.pt"
        expected_paths.add(relative)
        row = manifest.get((geometry, case_id), {})
        if row.get("path") != relative:
            fail(errors, f"manifest path mismatch: {geometry}/{case_id}")
        path = root / relative
        if not path.is_file():
            fail(errors, f"missing case file: {relative}")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size != int(row.get("source_bytes", -1)):
            fail(errors, f"case file size mismatch: {relative}")
    actual_paths = {
        p.relative_to(root).as_posix()
        for p in (root / "data").rglob("*") if p.is_file()
    }
    if actual_paths != expected_paths:
        fail(errors, "data directory membership differs from the manifest")
    return {"layout": "individual-case-files", "case_files": len(actual_paths),
            "case_bytes": total_bytes, "archives": 0}


def validate_checksums(root: Path, errors: list[str]) -> dict:
    path = root / "checksums.sha256"
    checked = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                fail(errors, f"invalid checksum line {line_number}")
                continue
            target = root / relative
            if not target.is_file():
                fail(errors, f"checksummed file missing: {relative}")
                continue
            if sha256_file(target) != expected:
                fail(errors, f"file checksum mismatch: {relative}")
            checked += 1
            if target.suffix == ".zip":
                print(f"[file checksum] {relative}")
    return {"checked_files": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    started = time.time()
    errors: list[str] = []
    split = validate_split(root, errors)
    manifest = load_manifest(root, errors)
    archives = validate_case_files(root, manifest, errors)
    geometry_reports = {}

    for geometry, spec in SPECS.items():
        mesh_report = validate_mesh(root, geometry, spec, errors)
        materials: Counter[str] = Counter()
        total_bytes = 0
        for index in range(1, 501):
            case_id = f"case{index:03d}"
            row = manifest.get((geometry, case_id))
            expected_digest = row["source_sha256"] if row else None
            case_report = validate_case(
                root, geometry, case_id, spec, expected_digest, errors
            )
            total_bytes += case_report["bytes"]
            materials[case_report["material"]] += 1
            if index % 50 == 0:
                print(f"[{geometry}] validated {index}/500")
        geometry_reports[geometry] = {
            **mesh_report,
            "cases": 500,
            "source_case_bytes": total_bytes,
            "materials": dict(sorted(materials.items())),
        }

    checksum_report = (
        validate_checksums(root, errors) if args.verify_checksums else {"skipped": True}
    )
    report = {
        "dataset": "Automotive Underbody Panel Impact Dataset",
        "version": "1.1.1",
        "distribution": "git-cases-v1",
        "status": "passed" if not errors else "failed",
        "cases_validated": 1500,
        "split_sizes": {key: len(value) for key, value in split.items()},
        "archives": archives,
        "geometries": geometry_reports,
        "checksums": checksum_report,
        "elapsed_seconds": time.time() - started,
        "errors": errors,
        "scope_note": (
            "Technical package validation only; provenance and remaining "
            "configuration limitations are documented in the release metadata."
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered)
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
