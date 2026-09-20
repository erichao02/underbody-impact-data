from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from prepare_floorfrontR_training_data import (
    DEFAULT_PANEL_KEY,
    PANEL_PID,
    build_graph,
    parse_key_mesh,
    read_curveplot_txt,
)


DEFAULT_CASE_ROOT = Path("cases_floorfrontR_random_50")
DEFAULT_OUT_ROOT = Path("training_data_floorfrontR_compact")
RESULTS_DIR = "results_compact"
NODE_RESULT_NAMES = ("x_displacement", "y_displacement", "z_displacement")
ELEMENT_RESULT_NAMES = ("effective_stress", "effective_plastic_strain")
MIN_UNIQUE_ELEMENT_ID_FRACTION = 0.99
MAX_DOMINANT_STRESS_CURVE_FRACTION = 0.95


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Convert compact floorfrontR LS-PrePost curve outputs to lightweight .pt training data."
    )
    ap.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    ap.add_argument("--sweep-root", type=Path, default=None, help="Process every velocity folder under this root.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--panel-key", type=Path, default=DEFAULT_PANEL_KEY)
    ap.add_argument("--panel-pid", type=int, default=PANEL_PID)
    ap.add_argument("--start", type=int, default=1, help="First case number to convert.")
    ap.add_argument("--end", type=int, default=0, help="Last case number to convert. Use 0 for all cases.")
    ap.add_argument("--stride", type=int, default=10, help="Save one state every N time steps.")
    ap.add_argument("--no-include-last", action="store_true", help="Do not append the final time state.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--expected-cases",
        type=int,
        default=0,
        help="Expected number of manifest/case entries. Use 0 to disable this check.",
    )
    ap.add_argument(
        "--require-all",
        action="store_true",
        help="Fail if any selected case is missing required results_compact files.",
    )
    ap.add_argument(
        "--delete-raw",
        action="store_true",
        help="Delete results_compact text files after a case .pt is successfully written.",
    )
    return ap.parse_args()


def save_npz_from_graph(graph: Dict[str, torch.Tensor], out_path: Path) -> None:
    arrays = {
        "nid": graph["nid"].numpy(),
        "pos": graph["pos"].numpy(),
        "edge_index": graph["edge_index"].numpy(),
        "boundary_mask": graph["boundary_mask"].numpy(),
    }
    if "element_id" in graph:
        arrays["element_id"] = graph["element_id"].numpy()
    np.savez_compressed(out_path, **arrays)


def make_stride_indices(n_steps: int, stride: int, include_last: bool) -> torch.Tensor:
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    idx = list(range(0, n_steps, stride))
    if include_last and idx[-1] != n_steps - 1:
        idx.append(n_steps - 1)
    return torch.tensor(idx, dtype=torch.long)


def read_manifest_rows(case_root: Path) -> List[dict]:
    case_manifest = case_root / "case_manifest.csv"
    if case_manifest.exists():
        with case_manifest.open("r", newline="") as f:
            return list(csv.DictReader(f))

    velocity_manifest = case_root.parent / "velocity_manifest.csv"
    if velocity_manifest.exists():
        with velocity_manifest.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        return [r for r in rows if r.get("velocity_set") == case_root.name]

    rows = []
    for case_dir in sorted(case_root.glob("case*")):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        rows.append(
            {
                "case": case_name,
                "key_file": str(case_dir / f"{case_name}.key"),
                "panel_pid": str(PANEL_PID),
            }
        )
    return rows


def filter_case_rows(rows: List[dict], start: int, end: int) -> List[dict]:
    if start <= 1 and end <= 0:
        return rows

    selected = []
    for row in rows:
        case_name = row.get("case", "")
        if not case_name.startswith("case"):
            selected.append(row)
            continue
        try:
            case_num = int(case_name[4:])
        except ValueError:
            selected.append(row)
            continue
        if case_num < start:
            continue
        if end > 0 and case_num > end:
            continue
        selected.append(row)
    return selected


def format_case_preview(cases: List[str], limit: int = 20) -> str:
    preview = ", ".join(sorted(cases)[:limit])
    if len(cases) > limit:
        preview += f", ... (+{len(cases) - limit} more)"
    return preview


def find_missing_compact_results(rows: List[dict]) -> List[str]:
    missing = []
    required_names = NODE_RESULT_NAMES + ELEMENT_RESULT_NAMES
    for row in rows:
        key_file = Path(row.get("key_file", ""))
        results_dir = key_file.parent / RESULTS_DIR
        if not results_dir.exists():
            missing.append(row["case"])
            continue
        if any(not (results_dir / name).exists() or (results_dir / name).stat().st_size <= 0 for name in required_names):
            missing.append(row["case"])
    return missing


def tensor3(values: List[Optional[str]]) -> Optional[torch.Tensor]:
    if any(v in (None, "") for v in values):
        return None
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def map_nodal_curves(
    graph_nid: torch.Tensor,
    curve_nid: torch.Tensor,
    curves: Dict[str, torch.Tensor],
    step_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nid_to_row = {int(n): i for i, n in enumerate(curve_nid.tolist())}
    order = []
    valid = []
    for n in graph_nid.tolist():
        idx = nid_to_row.get(int(n))
        if idx is None:
            order.append(-1)
            valid.append(False)
        else:
            order.append(idx)
            valid.append(True)

    valid_mask = torch.tensor(valid, dtype=torch.bool)
    valid_order = [idx for idx in order if idx >= 0]
    disp = torch.full((graph_nid.numel(), step_idx.numel(), 3), float("nan"), dtype=torch.float32)
    disp[valid_mask] = torch.stack(
        [
            curves["x_displacement"][valid_order][:, step_idx],
            curves["y_displacement"][valid_order][:, step_idx],
            curves["z_displacement"][valid_order][:, step_idx],
        ],
        dim=-1,
    )
    return disp, valid_mask


def fill_missing_nodal_disp(
    disp: torch.Tensor,
    valid_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    filled_mask = ~valid_mask
    if not bool(filled_mask.any()):
        return disp, filled_mask

    adjacency: Dict[int, List[int]] = {}
    for src, dst in edge_index.t().tolist():
        adjacency.setdefault(int(src), []).append(int(dst))

    for node_idx in torch.nonzero(filled_mask, as_tuple=False).flatten().tolist():
        neighbors = [idx for idx in adjacency.get(int(node_idx), []) if bool(valid_mask[idx])]
        if not neighbors:
            continue
        disp[node_idx] = disp[neighbors].mean(dim=0)
        valid_mask[node_idx] = True

    return disp, filled_mask


def repair_ids_by_order(reference_ids: torch.Tensor, curve_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Repair LS-PrePost curve-header duplicates when row order is anchored."""
    if reference_ids.numel() != curve_ids.numel():
        return curve_ids, 0
    if torch.equal(reference_ids, curve_ids):
        return curve_ids, 0

    mismatched = reference_ids != curve_ids
    mismatch_count = int(mismatched.sum())
    if mismatch_count == 0:
        return curve_ids, 0

    match_ratio = 1.0 - mismatch_count / max(1, reference_ids.numel())
    if match_ratio >= 0.99:
        return reference_ids.clone(), mismatch_count

    # LS-PrePost can label a long contiguous block of otherwise correctly
    # ordered curves with one repeated element ID. Accept that pattern only
    # when every singleton label agrees with the reference at its position
    # and singleton anchors exist on both sides of the mismatched block.
    unique_ids, inverse, counts = torch.unique(
        curve_ids, sorted=False, return_inverse=True, return_counts=True
    )
    singleton = counts[inverse] == 1
    reference_set = set(reference_ids.tolist())
    observed_set = set(unique_ids.tolist())
    mismatch_indices = torch.nonzero(mismatched, as_tuple=False).flatten()
    trusted_singleton = singleton & (curve_ids == reference_ids)
    singleton_count = int(singleton.sum())
    singleton_mismatch_count = int((singleton & mismatched).sum())
    allowed_singleton_mismatches = max(1, int(0.01 * singleton_count))
    trusted_singleton_indices = torch.nonzero(
        trusted_singleton, as_tuple=False
    ).flatten()
    anchored_before = bool(
        trusted_singleton_indices.numel()
        and (trusted_singleton_indices < mismatch_indices.min()).any()
    )
    anchored_after = bool(
        trusted_singleton_indices.numel()
        and (trusted_singleton_indices > mismatch_indices.max()).any()
    )
    if (
        observed_set.issubset(reference_set)
        and singleton_mismatch_count <= allowed_singleton_mismatches
        and anchored_before
        and anchored_after
    ):
        return reference_ids.clone(), mismatch_count

    return curve_ids, 0


def make_sample(
    row: dict,
    graph: Dict[str, torch.Tensor],
    fixed_mask: torch.Tensor,
    stride: int,
    include_last: bool,
) -> Optional[dict]:
    case_name = row["case"]
    key_file = Path(row.get("key_file", ""))
    if not key_file.is_absolute():
        case_dir = key_file.parent
    else:
        case_dir = key_file.parent
    if not case_dir.exists():
        return None

    results_dir = case_dir / RESULTS_DIR
    if not results_dir.exists():
        return None

    required = [results_dir / name for name in NODE_RESULT_NAMES + ELEMENT_RESULT_NAMES]
    if not all(path.exists() for path in required):
        return None

    node_curves = {}
    node_time = None
    node_nid = None
    for name in NODE_RESULT_NAMES:
        time, nid, values = read_curveplot_txt(results_dir / name)
        if node_time is None:
            node_time = time
            node_nid = nid
        elif not torch.allclose(node_time, time):
            raise RuntimeError(f"Node time grid mismatch in {case_dir}")
        if not torch.equal(node_nid, nid):
            raise RuntimeError(f"Node ID order mismatch in {case_dir}")
        node_curves[name] = values

    node_step_idx = make_stride_indices(node_time.numel(), stride, include_last)
    disp, valid_node_mask = map_nodal_curves(graph["nid"], node_nid, node_curves, node_step_idx)
    raw_valid_node_mask = valid_node_mask.clone()
    disp, _ = fill_missing_nodal_disp(disp, valid_node_mask.clone(), graph["edge_index"])
    valid_node_mask = torch.isfinite(disp).flatten(start_dim=1).all(dim=1)
    filled_node_mask = valid_node_mask & ~raw_valid_node_mask

    element_curves = {}
    element_time = None
    element_id = None
    for name in ELEMENT_RESULT_NAMES:
        time, ids, values = read_curveplot_txt(results_dir / name)
        if element_time is None:
            element_time = time
            element_id = ids
        elif not torch.allclose(element_time, time):
            raise RuntimeError(f"Element time grid mismatch in {case_dir}")
        if not torch.equal(element_id, ids):
            raise RuntimeError(f"Element ID order mismatch in {case_dir}")
        element_curves[name] = values

    stress_curves = element_curves["effective_stress"]
    unique_id_fraction = float(element_id.unique().numel()) / max(1, element_id.numel())
    _, stress_curve_counts = torch.unique(stress_curves, dim=0, return_counts=True)
    dominant_stress_fraction = float(stress_curve_counts.max()) / max(
        1, stress_curves.shape[0]
    )
    if unique_id_fraction < MIN_UNIQUE_ELEMENT_ID_FRACTION:
        raise RuntimeError(
            f"Degenerate compact stress element IDs in {case_dir}: "
            f"unique_fraction={unique_id_fraction:.6f}"
        )
    if dominant_stress_fraction > MAX_DOMINANT_STRESS_CURVE_FRACTION:
        raise RuntimeError(
            f"Degenerate compact stress curves in {case_dir}: "
            f"dominant_fraction={dominant_stress_fraction:.6f}"
        )

    element_id, repaired_element_ids = repair_ids_by_order(graph["element_id"], element_id)
    elem_step_idx = make_stride_indices(element_time.numel(), stride, include_last)
    effective_stress = element_curves["effective_stress"][:, elem_step_idx]
    effective_plastic_strain = element_curves["effective_plastic_strain"][:, elem_step_idx]

    impact_xyz = tensor3([row.get("impact_x"), row.get("impact_y"), row.get("impact_z")])
    ball_center_xyz = tensor3([row.get("ball_center_x"), row.get("ball_center_y"), row.get("ball_center_z")])
    velocity_vz = row.get("velocity_vz", "")
    if velocity_vz in ("", None):
        velocity_vz = "3464.1"
    velocity_vx = float(row.get("velocity_vx") or 0.0)
    velocity_vy = float(row.get("velocity_vy") or 0.0)
    velocity_vz_value = float(velocity_vz)
    impact_speed = float(
        row.get("impact_speed")
        or (velocity_vx**2 + velocity_vy**2 + velocity_vz_value**2) ** 0.5
    )

    sample = {
        "case": case_name,
        "panel_pid": torch.tensor(int(row.get("panel_pid") or PANEL_PID), dtype=torch.long),
        "nid": graph["nid"],
        "boundary_mask": fixed_mask,
        "valid_node_mask": valid_node_mask,
        "raw_valid_node_mask": raw_valid_node_mask,
        "filled_node_mask": filled_node_mask,
        "filled_node_count": torch.tensor(int(filled_node_mask.sum()), dtype=torch.long),
        "time": node_time[node_step_idx],
        "time_indices": node_step_idx,
        "disp": disp,
        "disp_z": disp[..., 2],
        "element_time": element_time[elem_step_idx],
        "element_time_indices": elem_step_idx,
        "element_id": element_id,
        "repaired_element_ids": torch.tensor(repaired_element_ids, dtype=torch.long),
        "effective_stress": effective_stress,
        "effective_plastic_strain": effective_plastic_strain,
        "effective_strain": effective_plastic_strain,
        "velocity_vz": torch.tensor(velocity_vz_value, dtype=torch.float32),
        "velocity_xyz": torch.tensor(
            [velocity_vx, velocity_vy, velocity_vz_value], dtype=torch.float32
        ),
        "impact_speed": torch.tensor(impact_speed, dtype=torch.float32),
    }

    if impact_xyz is not None:
        sample["impact_xyz"] = impact_xyz
        sample["impact_node_distance"] = torch.linalg.norm(graph["pos"] - impact_xyz[None, :], dim=1)
    if ball_center_xyz is not None:
        sample["ball_center_xyz"] = ball_center_xyz
    if row.get("impact_element_id"):
        sample["impact_element_id"] = torch.tensor(int(row["impact_element_id"]), dtype=torch.long)
    if row.get("velocity_set"):
        sample["velocity_set"] = row["velocity_set"]

    lhs_fields = (
        "mass_ratio",
        "impactor_mass",
        "impactor_density",
        "theta_deg",
        "phi_deg",
        "material_young_mpa",
        "material_poisson",
    )
    for field in lhs_fields:
        if row.get(field) not in ("", None):
            sample[field] = torch.tensor(float(row[field]), dtype=torch.float32)
    if row.get("material_index") not in ("", None):
        material_index = int(row["material_index"])
        sample["material_index"] = torch.tensor(material_index, dtype=torch.long)
        sample["material_one_hot"] = torch.nn.functional.one_hot(
            torch.tensor(material_index), num_classes=3
        ).to(torch.float32)
    if row.get("material_name"):
        sample["material_name"] = row["material_name"]

    condition_names = (
        "impact_x",
        "impact_y",
        "impact_speed",
        "mass_ratio",
        "theta_deg",
        "phi_deg",
        "material_index",
    )
    if all(row.get(name) not in ("", None) for name in condition_names):
        sample["condition_names"] = list(condition_names)
        sample["condition_vector"] = torch.tensor(
            [float(row[name]) for name in condition_names], dtype=torch.float32
        )

    return sample


def convert_case(
    row: dict,
    graph: Dict[str, torch.Tensor],
    fixed_mask: torch.Tensor,
    out_cases_dir: Path,
    stride: int,
    include_last: bool,
    overwrite: bool,
    delete_raw: bool,
) -> Tuple[str, str, str]:
    case_name = row["case"]
    out_path = out_cases_dir / f"{case_name}.pt"
    if out_path.exists() and not overwrite:
        return "skipped_existing", case_name, str(out_path)

    try:
        sample = make_sample(row, graph, fixed_mask, stride, include_last)
        if sample is None:
            return "skipped_missing", case_name, ""

        tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
        torch.save(sample, tmp_path)
        tmp_path.replace(out_path)

        if delete_raw:
            key_file = Path(row.get("key_file", ""))
            results_dir = key_file.parent / RESULTS_DIR
            for name in NODE_RESULT_NAMES + ELEMENT_RESULT_NAMES:
                path = results_dir / name
                if path.exists():
                    path.unlink()

        return "converted", case_name, str(out_path)
    except Exception as exc:
        return "error", case_name, str(exc)


def write_readme(out_root: Path, case_root: Path, stride: int, include_last: bool) -> None:
    text = f"""# compact floorfrontR training data

Source case root: `{case_root}`

This dataset stores only the reduced training targets:

- nodal displacement: `disp`, shape `(N, T_reduced, 3)`
- shell effective stress: `effective_stress`, shape `(Ne, T_reduced)`
- shell effective strain alias: `effective_strain`, shape `(Ne, T_reduced)`
- original LS-PrePost variable: `effective_plastic_strain`, shape `(Ne, T_reduced)`

Time downsampling:

- stride: `{stride}`
- include final state: `{include_last}`

The full LS-PrePost text curves are read from each case `results_compact/` folder.
The saved `.pt` files keep only the downsampled time states.
"""
    (out_root / "README.md").write_text(text, newline="")


def convert_case_root(
    case_root: Path,
    base_out_root: Path,
    graph: Dict[str, torch.Tensor],
    fixed_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[Path, int, int, int, int]:
    out_root = base_out_root / case_root.name
    out_cases = out_root / "cases"
    out_cases.mkdir(parents=True, exist_ok=True)

    mesh_graph_pt = out_root / "mesh_graph.pt"
    if args.overwrite or not mesh_graph_pt.exists():
        torch.save(graph, mesh_graph_pt)
        save_npz_from_graph(graph, out_root / "mesh_graph.npz")

    all_rows = read_manifest_rows(case_root)
    if args.expected_cases > 0 and len(all_rows) != args.expected_cases:
        raise RuntimeError(f"Expected {args.expected_cases} cases from {case_root}, found {len(all_rows)}.")
    rows = filter_case_rows(all_rows, args.start, args.end)
    missing_before = find_missing_compact_results(rows)
    if missing_before and args.require_all:
        raise RuntimeError(
            f"Missing compact export for {len(missing_before)} case(s): "
            f"{format_case_preview(missing_before)}"
        )

    conditions = {
        "case": [r["case"] for r in all_rows],
        "velocity_vz": torch.tensor(
            [float(r.get("velocity_vz") or 3464.1) for r in all_rows],
            dtype=torch.float32,
        ),
    }
    optional_float_fields = (
        "impact_x",
        "impact_y",
        "impact_z",
        "velocity_vx",
        "velocity_vy",
        "impact_speed",
        "mass_ratio",
        "impactor_mass",
        "impactor_density",
        "theta_deg",
        "phi_deg",
        "material_young_mpa",
        "material_poisson",
    )
    for field in optional_float_fields:
        if all(row.get(field) not in ("", None) for row in all_rows):
            conditions[field] = torch.tensor(
                [float(row[field]) for row in all_rows], dtype=torch.float32
            )
    if all(row.get("material_index") not in ("", None) for row in all_rows):
        conditions["material_index"] = torch.tensor(
            [int(row["material_index"]) for row in all_rows], dtype=torch.long
        )
    if all(row.get("material_name") for row in all_rows):
        conditions["material_name"] = [row["material_name"] for row in all_rows]
    torch.save(conditions, out_root / "case_conditions.pt")
    write_readme(out_root, case_root, args.stride, not args.no_include_last)

    converted = 0
    skipped_missing = 0
    missing_cases = []
    skipped_existing = 0
    errors = []
    workers = max(1, int(args.workers))

    if workers == 1:
        iterator = [
            convert_case(
                row,
                graph,
                fixed_mask,
                out_cases,
                args.stride,
                not args.no_include_last,
                args.overwrite,
                args.delete_raw,
            )
            for row in rows
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    convert_case,
                    row,
                    graph,
                    fixed_mask,
                    out_cases,
                    args.stride,
                    not args.no_include_last,
                    args.overwrite,
                    args.delete_raw,
                )
                for row in rows
            ]
            iterator = [future.result() for future in as_completed(futures)]

    for status, case_name, message in iterator:
        if status == "converted":
            converted += 1
        elif status == "skipped_missing":
            skipped_missing += 1
            missing_cases.append(case_name)
        elif status == "skipped_existing":
            skipped_existing += 1
        else:
            errors.append((case_name, message))

    if errors:
        detail = "\n".join(f"  {case}: {msg}" for case, msg in errors[:10])
        raise RuntimeError(f"Failed converting {len(errors)} cases under {case_root}:\n{detail}")
    if missing_cases:
        print(f"[INFO] missing compact result cases: {format_case_preview(missing_cases)}")

    return out_root, converted, skipped_existing, skipped_missing, len(rows)


def main() -> None:
    args = parse_args()

    nodes, elements, fixed_nodes = parse_key_mesh(args.panel_key, args.panel_pid)
    graph = build_graph(nodes, elements)
    fixed_set = set(fixed_nodes)
    fixed_mask = torch.tensor([int(n) in fixed_set for n in graph["nid"].tolist()], dtype=torch.bool)
    graph["boundary_mask"] = fixed_mask

    if args.sweep_root is not None:
        roots = sorted([p for p in args.sweep_root.iterdir() if p.is_dir() and p.name.startswith("v")])
    else:
        roots = [args.case_root]

    print(f"[OK] graph nodes: {graph['nid'].numel()}")
    print(f"[OK] graph directed edges: {graph['edge_index'].shape[1]}")
    print(f"[OK] fixed boundary nodes: {int(fixed_mask.sum())}")
    print(f"[OK] time stride: {args.stride}")

    for root in roots:
        out_root, converted, skipped_existing, skipped_missing, total = convert_case_root(
            root, args.out_root, graph, fixed_mask, args
        )
        print("=========================================")
        print(f"[OK] source: {root}")
        print(f"[OK] output: {out_root}")
        print(f"[OK] manifest cases: {total}")
        print(f"[OK] converted: {converted}")
        print(f"[OK] skipped existing: {skipped_existing}")
        print(f"[OK] skipped missing compact results: {skipped_missing}")


if __name__ == "__main__":
    main()
