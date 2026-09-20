from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


PANEL_PID = 2000395
DEFAULT_PANEL_KEY = Path("floor_panel_largest_2_pid_2000395_133_floorfrontR.key")
DEFAULT_CASE_ROOT = Path("cases_floorfrontR_random_50")
DEFAULT_OUT_ROOT = Path("training_data_floorfrontR_random_50")
NODE_COORD_NAMES = ("x_coordinate", "y_coordinate", "z_coordinate")
NODE_RESULT_NAMES = ("x_displacement", "y_displacement", "z_displacement")
ELEMENT_RESULT_NAMES = (
    "x_stress",
    "y_stress",
    "z_stress",
    "xy_stress",
    "yz_stress",
    "zx_stress",
    "effective_plastic_strain",
    "pressure",
    "effective_stress",
    "lower_Ipt_x_strain",
    "lower_Ipt_y_strain",
    "lower_Ipt_z_strain",
    "lower_Ipt_xy_strain",
    "lower_Ipt_yz_strain",
    "lower_Ipt_zx_strain",
    "upper_Ipt_x_strain",
    "upper_Ipt_y_strain",
    "upper_Ipt_z_strain",
    "upper_Ipt_xy_strain",
    "upper_Ipt_yz_strain",
    "upper_Ipt_zx_strain",
)


@dataclass
class ShellElement:
    eid: int
    pid: int
    nodes: Tuple[int, ...]


def is_keyword(line: str) -> bool:
    return line.lstrip().startswith("*")


def split_fields(line: str) -> List[str]:
    return line.replace(",", " ").split()


def parse_key_mesh(path: Path, panel_pid: int) -> Tuple[Dict[int, Tuple[float, float, float]], List[ShellElement], List[int]]:
    nodes: Dict[int, Tuple[float, float, float]] = {}
    elements: List[ShellElement] = []
    fixed_nodes: List[int] = []

    mode: Optional[str] = None
    pending_spc_set = False

    with path.open("r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("$"):
                continue

            upper = line.upper()
            if is_keyword(line):
                if upper.startswith("*NODE"):
                    mode = "node"
                elif upper.startswith("*ELEMENT_SHELL"):
                    mode = "shell"
                elif upper.startswith("*SET_NODE_LIST"):
                    mode = "node_set"
                    pending_spc_set = False
                elif upper.startswith("*BOUNDARY_SPC_SET"):
                    mode = "spc"
                    pending_spc_set = True
                else:
                    mode = None
                continue

            if mode == "node":
                parts = split_fields(line)
                if len(parts) >= 4:
                    try:
                        nid = int(parts[0])
                        nodes[nid] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    except ValueError:
                        pass

            elif mode == "shell":
                parts = split_fields(line)
                if len(parts) >= 6:
                    try:
                        eid = int(parts[0])
                        pid = int(parts[1])
                        elem_nodes = [int(x) for x in parts[2:6]]
                    except ValueError:
                        continue
                    if pid != panel_pid:
                        continue
                    if elem_nodes[3] == 0 or elem_nodes[3] == elem_nodes[2]:
                        elem_nodes = elem_nodes[:3]
                    elements.append(ShellElement(eid=eid, pid=pid, nodes=tuple(elem_nodes)))

            elif mode == "spc" and pending_spc_set:
                # The first non-comment data line of *BOUNDARY_SPC_SET gives the fixed node set ID.
                # The actual nodes are read from the following *SET_NODE_LIST block in this deck.
                pending_spc_set = False

            elif mode == "node_set":
                parts = split_fields(line)
                if len(parts) == 1:
                    # set ID line
                    continue
                for token in parts:
                    try:
                        nid = int(token)
                    except ValueError:
                        continue
                    if nid > 0:
                        fixed_nodes.append(nid)

    if not nodes:
        raise RuntimeError(f"No *NODE data found in {path}")
    if not elements:
        raise RuntimeError(f"No shell elements with PID={panel_pid} found in {path}")

    return nodes, elements, sorted(set(fixed_nodes))


def build_graph(
    nodes: Dict[int, Tuple[float, float, float]],
    elements: Iterable[ShellElement],
) -> Dict[str, torch.Tensor]:
    elements = list(elements)
    used_nids = sorted({nid for e in elements for nid in e.nodes})
    nid_to_idx = {nid: i for i, nid in enumerate(used_nids)}

    pos = torch.tensor([nodes[nid] for nid in used_nids], dtype=torch.float32)
    nid = torch.tensor(used_nids, dtype=torch.long)
    element_id = torch.tensor([e.eid for e in elements], dtype=torch.long)

    undirected_edges = set()
    for e in elements:
        ns = e.nodes
        for i, a in enumerate(ns):
            b = ns[(i + 1) % len(ns)]
            if a in nid_to_idx and b in nid_to_idx:
                ia, ib = nid_to_idx[a], nid_to_idx[b]
                undirected_edges.add((min(ia, ib), max(ia, ib)))

    directed = []
    for ia, ib in sorted(undirected_edges):
        directed.append((ia, ib))
        directed.append((ib, ia))

    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    return {"nid": nid, "pos": pos, "edge_index": edge_index, "element_id": element_id}


def read_manifest(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def read_curveplot_txt(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nids = []
    curves = []
    time_ref = None

    lines = path.read_text(errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and line[0].isdigit() and "#pts=" in line:
            nid = int(line.split()[0])
            nids.append(nid)
            i += 3
            ts, vs = [], []
            while i < len(lines):
                row = lines[i].strip()
                if row.lower().startswith("endcurve"):
                    break
                parts = row.split()
                if len(parts) == 2:
                    ts.append(float(parts[0]))
                    vs.append(float(parts[1]))
                i += 1
            t = np.asarray(ts, dtype=np.float32)
            v = np.asarray(vs, dtype=np.float32)
            if time_ref is None:
                time_ref = t
            elif not np.allclose(time_ref, t):
                raise ValueError(f"Time grid mismatch in {path}")
            curves.append(v)
        i += 1

    if time_ref is None:
        raise RuntimeError(f"No node curves found in {path}")
    return (
        torch.tensor(time_ref, dtype=torch.float32),
        torch.tensor(nids, dtype=torch.long),
        torch.tensor(np.stack(curves, axis=0), dtype=torch.float32),
    )


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


def make_case_sample(
    row: dict,
    graph: Dict[str, torch.Tensor],
    fixed_mask: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    key_path = Path(row["key_file"])
    case_dir = key_path.parent
    results_dir = case_dir / "results"
    if not results_dir.exists():
        return None

    node_paths = [results_dir / name for name in NODE_RESULT_NAMES]
    if not all(path.exists() for path in node_paths):
        return None

    node_curves = {}
    node_time = None
    label_nid = None
    for name, path in zip(NODE_RESULT_NAMES, node_paths):
        time, curve_nid, values = read_curveplot_txt(path)
        if node_time is None:
            node_time = time
            label_nid = curve_nid
        elif not torch.allclose(node_time, time):
            raise RuntimeError(f"Node time grid mismatch in {case_dir}")
        if not torch.equal(label_nid, curve_nid):
            raise RuntimeError(f"Node ID order mismatch in {case_dir}")
        node_curves[name] = values

    nid_to_row = {int(n): i for i, n in enumerate(label_nid.tolist())}
    order = []
    valid_node_mask = []
    for n in graph["nid"].tolist():
        idx = nid_to_row.get(int(n))
        if idx is None:
            order.append(-1)
            valid_node_mask.append(False)
        else:
            order.append(idx)
            valid_node_mask.append(True)

    impact_xyz = torch.tensor(
        [float(row["impact_x"]), float(row["impact_y"]), float(row["impact_z"])],
        dtype=torch.float32,
    )
    ball_center = torch.tensor(
        [float(row["ball_center_x"]), float(row["ball_center_y"]), float(row["ball_center_z"])],
        dtype=torch.float32,
    )
    impact_features = torch.tensor(
        [
            float(row["impact_x"]),
            float(row["impact_y"]),
            float(row["impact_z"]),
            float(row["ball_center_x"]),
            float(row["ball_center_y"]),
            float(row["ball_center_z"]),
            3464.1,
            float(row["min_boundary_distance_mm"]),
        ],
        dtype=torch.float32,
    )
    distance = torch.linalg.norm(graph["pos"] - impact_xyz[None, :], dim=1)

    valid_node_mask = torch.tensor(valid_node_mask, dtype=torch.bool)
    valid_order = [idx for idx in order if idx >= 0]
    disp = torch.full((graph["nid"].numel(), node_time.numel(), 3), float("nan"), dtype=torch.float32)
    disp[valid_node_mask] = torch.stack(
        [
            node_curves["x_displacement"][valid_order],
            node_curves["y_displacement"][valid_order],
            node_curves["z_displacement"][valid_order],
        ],
        dim=-1,
    )

    sample = {
        "case": row["case"],
        "panel_pid": torch.tensor(int(row["panel_pid"]), dtype=torch.long),
        "impact_element_id": torch.tensor(int(row["impact_element_id"]), dtype=torch.long),
        "impact_xyz": impact_xyz,
        "ball_center_xyz": ball_center,
        "impact_features": impact_features,
        "impact_node_distance": distance,
        "boundary_mask": fixed_mask,
        "valid_node_mask": valid_node_mask,
        "time": node_time,
        "nid": graph["nid"],
        "disp": disp,
        "disp_z": disp[..., 2],
    }

    coord_paths = [results_dir / name for name in NODE_COORD_NAMES]
    if all(path.exists() for path in coord_paths):
        coord_curves = {}
        coord_time = None
        coord_nid = None
        for name, path in zip(NODE_COORD_NAMES, coord_paths):
            time, curve_nid, values = read_curveplot_txt(path)
            if coord_time is None:
                coord_time = time
                coord_nid = curve_nid
            elif not torch.allclose(coord_time, time):
                raise RuntimeError(f"Coordinate time grid mismatch in {case_dir}")
            if not torch.equal(coord_nid, curve_nid):
                raise RuntimeError(f"Coordinate node ID order mismatch in {case_dir}")

            coord_curves[name] = values

        coord_nid_to_row = {int(n): i for i, n in enumerate(coord_nid.tolist())}
        coord_order = []
        coord_valid_mask = []
        for n in graph["nid"].tolist():
            idx = coord_nid_to_row.get(int(n))
            if idx is None:
                coord_order.append(-1)
                coord_valid_mask.append(False)
            else:
                coord_order.append(idx)
                coord_valid_mask.append(True)
        coord_valid_mask = torch.tensor(coord_valid_mask, dtype=torch.bool)
        coord_valid_order = [idx for idx in coord_order if idx >= 0]
        coord = torch.full((graph["nid"].numel(), coord_time.numel(), 3), float("nan"), dtype=torch.float32)
        coord[coord_valid_mask] = torch.stack(
            [
                coord_curves["x_coordinate"][coord_valid_order],
                coord_curves["y_coordinate"][coord_valid_order],
                coord_curves["z_coordinate"][coord_valid_order],
            ],
            dim=-1,
        )
        sample["coord_time"] = coord_time
        sample["coord"] = coord
        sample["valid_coord_mask"] = coord_valid_mask

    element_results = {}
    element_ids = None
    element_time = None
    for name in ELEMENT_RESULT_NAMES:
        path = results_dir / name
        if not path.exists():
            continue
        time, ids, values = read_curveplot_txt(path)
        if element_time is None:
            element_time = time
            element_ids = ids
        elif not torch.allclose(element_time, time):
            raise RuntimeError(f"Element time grid mismatch in {case_dir}")
        if not torch.equal(element_ids, ids):
            raise RuntimeError(f"Element ID order mismatch in {case_dir}")
        element_results[name] = values

    if element_results:
        sample["element_time"] = element_time
        sample["element_id"] = element_ids
        sample["element_results"] = element_results

    return sample


def convert_case_to_pt(
    row: dict,
    graph: Dict[str, torch.Tensor],
    fixed_mask: torch.Tensor,
    out_cases_dir: Path,
    overwrite: bool,
) -> Tuple[str, str, str]:
    case_name = row["case"]
    out_path = out_cases_dir / f"{case_name}.pt"

    if out_path.exists() and not overwrite:
        return "skipped_existing", case_name, str(out_path)

    try:
        sample = make_case_sample(row, graph, fixed_mask)
        if sample is None:
            return "skipped_missing", case_name, ""
        tmp_path = out_path.with_name(f"{out_path.name}.tmp.{os.getpid()}")
        torch.save(sample, tmp_path)
        tmp_path.replace(out_path)
        return "converted", case_name, str(out_path)
    except Exception as exc:
        return "error", case_name, str(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-key", type=Path, default=DEFAULT_PANEL_KEY)
    ap.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--panel-pid", type=int, default=PANEL_PID)
    ap.add_argument("--workers", type=int, default=1, help="Number of cases to convert in parallel")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate existing .pt files")
    ap.add_argument("--skip-existing", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    manifest = args.case_root / "case_manifest.csv"
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "cases").mkdir(exist_ok=True)

    nodes, elements, fixed_nodes = parse_key_mesh(args.panel_key, args.panel_pid)
    graph = build_graph(nodes, elements)
    fixed_set = set(fixed_nodes)
    fixed_mask = torch.tensor([int(n) in fixed_set for n in graph["nid"].tolist()], dtype=torch.bool)
    graph["boundary_mask"] = fixed_mask

    mesh_graph_pt = args.out_root / "mesh_graph.pt"
    mesh_graph_npz = args.out_root / "mesh_graph.npz"
    if args.overwrite or not mesh_graph_pt.exists():
        torch.save(graph, mesh_graph_pt)
    if args.overwrite or not mesh_graph_npz.exists():
        save_npz_from_graph(graph, mesh_graph_npz)

    rows = read_manifest(manifest)
    case_conditions_pt = args.out_root / "case_conditions.pt"
    refresh_case_conditions = args.overwrite or not case_conditions_pt.exists()
    if not refresh_case_conditions:
        try:
            existing_conditions = torch.load(case_conditions_pt, map_location="cpu", weights_only=False)
            refresh_case_conditions = existing_conditions.get("case") != [r["case"] for r in rows]
        except Exception:
            refresh_case_conditions = True
    if refresh_case_conditions:
        torch.save(
            {
                "case": [r["case"] for r in rows],
                "impact_features": torch.tensor(
                    [
                        [
                            float(r["impact_x"]),
                            float(r["impact_y"]),
                            float(r["impact_z"]),
                            float(r["ball_center_x"]),
                            float(r["ball_center_y"]),
                            float(r["ball_center_z"]),
                            3464.1,
                            float(r["min_boundary_distance_mm"]),
                        ]
                        for r in rows
                    ],
                    dtype=torch.float32,
                ),
            },
            case_conditions_pt,
        )

    converted = 0
    skipped_missing = 0
    skipped_existing = 0
    errors = []
    out_cases_dir = args.out_root / "cases"
    workers = max(1, int(args.workers))

    if workers == 1:
        for row in rows:
            status, case_name, message = convert_case_to_pt(
                row, graph, fixed_mask, out_cases_dir, args.overwrite
            )
            if status == "converted":
                converted += 1
            elif status == "skipped_missing":
                skipped_missing += 1
            elif status == "skipped_existing":
                skipped_existing += 1
            else:
                errors.append((case_name, message))
    else:
        print(f"[INFO] parallel case conversion workers: {workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    convert_case_to_pt,
                    row,
                    graph,
                    fixed_mask,
                    out_cases_dir,
                    args.overwrite,
                )
                for row in rows
            ]
            for future in as_completed(futures):
                status, case_name, message = future.result()
                if status == "converted":
                    converted += 1
                    print(f"[OK] converted {case_name}")
                elif status == "skipped_missing":
                    skipped_missing += 1
                elif status == "skipped_existing":
                    skipped_existing += 1
                else:
                    errors.append((case_name, message))

    print(f"[OK] graph nodes: {graph['nid'].numel()}")
    print(f"[OK] graph directed edges: {graph['edge_index'].shape[1]}")
    print(f"[OK] fixed boundary nodes: {int(fixed_mask.sum())}")
    print(f"[OK] saved: {args.out_root / 'mesh_graph.pt'}")
    print(f"[OK] saved: {args.out_root / 'case_conditions.pt'}")
    print(f"[OK] converted case result files: {converted}")
    print(f"[OK] skipped existing case files: {skipped_existing}")
    print(f"[OK] skipped cases without displacement results: {skipped_missing}")
    if errors:
        details = "\n".join(f"  {case}: {message}" for case, message in errors[:10])
        raise RuntimeError(f"Case conversion failed for {len(errors)} case(s):\n{details}")


if __name__ == "__main__":
    main()
