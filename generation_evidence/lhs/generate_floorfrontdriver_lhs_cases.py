from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

import generate_floorfrontdriver_random_cases as base


TEMPLATE = Path("floor_panel_largest_3_pid_2000394_373_floorfrontdriver_with_ball.key")
OUTPUT_ROOT = Path("cases_floorfrontdriver_lhs_100")
DESIGN_NAME = "floorfrontdriver"
DEFAULT_CASES = 100
DEFAULT_SEED = 20260721
PANEL_PID = 2000394
BALL_PID = 9636
BALL_MID = 174
BASE_DENSITY = 5.205e-5
BASE_MASS = 0.01
BALL_GAP_MM = 5.0

SPEED_RANGE = (1732.05, 5196.15)
MASS_RATIO_RANGE = (0.75, 1.25)
THETA_RANGE_DEG = (0.0, 15.0)
PHI_RANGE_DEG = (0.0, 360.0)

MATERIALS = (
    {"index": 0, "name": "aluminum_rigid", "young_mpa": 70000.0, "poisson": 0.33},
    {"index": 1, "name": "titanium_rigid", "young_mpa": 110000.0, "poisson": 0.34},
    {"index": 2, "name": "steel_rigid", "young_mpa": 210000.0, "poisson": 0.30},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate constrained LHS {DESIGN_NAME} impact cases."
    )
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def lhs(n: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    design = np.empty((n, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        design[:, dimension] = (rng.permutation(n) + rng.random(n)) / n
    return design


def pairwise_min_distance(values: np.ndarray) -> float:
    delta = values[:, None, :] - values[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(distance2, np.inf)
    return float(np.sqrt(distance2.min()))


def map_xy_to_panel(
    design: np.ndarray,
    candidates: list[dict],
    rng: np.random.Generator,
) -> tuple[list[dict], np.ndarray]:
    centers = np.asarray([item["center"][:2] for item in candidates], dtype=np.float64)
    low = centers.min(axis=0)
    span = centers.max(axis=0) - low
    normalized = (centers - low) / span
    available = np.ones(len(candidates), dtype=bool)
    selected_indices = np.empty(len(design), dtype=np.int64)

    # Random order avoids systematically giving early rows the best projection.
    for row_index in rng.permutation(len(design)):
        distance2 = ((normalized - design[row_index, :2]) ** 2).sum(axis=1)
        distance2[~available] = np.inf
        chosen = int(np.argmin(distance2))
        selected_indices[row_index] = chosen
        available[chosen] = False

    mapped = design.copy()
    mapped[:, :2] = normalized[selected_indices]
    return [candidates[index] for index in selected_indices], mapped


def make_design(n: int, candidates: list[dict], seed: int, trials: int) -> tuple[np.ndarray, list[dict], float]:
    root_rng = np.random.default_rng(seed)
    best_design = None
    best_impacts = None
    best_score = -np.inf

    for _ in range(trials):
        trial_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
        rng = np.random.default_rng(trial_seed)
        design = lhs(n, 7, rng)
        impacts, mapped = map_xy_to_panel(design, candidates, rng)
        material_coordinate = (np.floor(mapped[:, 6] * len(MATERIALS)) + 0.5) / len(MATERIALS)
        scored = mapped.copy()
        scored[:, 6] = material_coordinate
        score = pairwise_min_distance(scored)
        if score > best_score:
            best_design = mapped
            best_impacts = impacts
            best_score = score

    assert best_design is not None and best_impacts is not None
    return best_design, best_impacts, best_score


def scale(unit_value: float, limits: tuple[float, float]) -> float:
    return limits[0] + unit_value * (limits[1] - limits[0])


def direction(theta_deg: float, phi_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    return np.asarray(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)],
        dtype=np.float64,
    )


def update_mat_rigid(block: list[str], density: float, young_mpa: float, poisson: float) -> tuple[list[str], bool]:
    output = list(block)
    for index, line in enumerate(output[1:], start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        fields = stripped.split()
        if len(fields) >= 4 and fields[0] == str(BALL_MID):
            output[index] = f"{BALL_MID:10d}{density:10.4E}{young_mpa:10.1f}{poisson:10.4f}\n"
            return output, True
    return output, False


def update_initial_velocity(block: list[str], velocity: np.ndarray) -> tuple[list[str], bool]:
    output = list(block)
    for index, line in enumerate(output[1:], start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        fields = stripped.split()
        if fields and fields[0] == str(BALL_PID):
            values = [BALL_PID, *velocity.tolist(), 0.0, 0.0, 0.0, 0]
            output[index] = (
                f"{values[0]:10d}"
                + "".join(f"{value:10.3E}" for value in values[1:7])
                + f"{values[7]:10d}\n"
            )
            return output, True
    return output, False


def make_case_key(
    template: dict,
    impact: dict,
    velocity: np.ndarray,
    density: float,
    material: dict,
) -> tuple[list[str], np.ndarray]:
    impact_center = np.asarray(impact["center"], dtype=np.float64)
    travel_direction = velocity / np.linalg.norm(velocity)
    ball_center = impact_center - travel_direction * (template["ball_radius"] + BALL_GAP_MM)
    output_blocks: list[list[str]] = []
    material_updated = False
    velocity_updated = False

    for block in template["blocks"]:
        if base.block_is_control_adaptive(block) or base.block_is_end(block):
            continue
        key = block[0].strip().upper()
        if key == "*NODE":
            modified = base.move_ball_nodes_in_node_block(
                block, template["ball_node_ids"], template["ball_center"], ball_center
            )
        elif key == "*PART":
            modified = base.update_part_adpopt_block(block, PANEL_PID, 0)
            modified = base.update_part_adpopt_block(modified, BALL_PID, 0)
        elif key.startswith("*MAT_RIGID"):
            modified, changed = update_mat_rigid(
                block, density, material["young_mpa"], material["poisson"]
            )
            material_updated |= changed
        elif key == "*INITIAL_VELOCITY_RIGID_BODY":
            modified, changed = update_initial_velocity(block, velocity)
            velocity_updated |= changed
        else:
            modified = block
        output_blocks.append(modified)

    if not material_updated:
        raise RuntimeError(f"Could not update rigid material MID {BALL_MID}")
    if not velocity_updated:
        raise RuntimeError(f"Could not update initial velocity for PID {BALL_PID}")

    lines = [line for block in output_blocks for line in block]
    lines.append("*END\n")
    return lines, ball_center


def min_impact_spacing(impacts: list[dict]) -> float:
    xy = np.asarray([item["center"][:2] for item in impacts])
    delta = xy[:, None, :] - xy[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(distance2, np.inf)
    return float(np.sqrt(distance2.min()))


def main() -> None:
    args = parse_args()
    if args.cases < 3:
        raise ValueError("--cases must be at least 3")
    if not args.template.exists():
        raise FileNotFoundError(args.template)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output root is not empty: {args.output_root}; use --overwrite to replace keys")

    template = base.parse_template(args.template)
    design, impacts, score = make_design(args.cases, template["candidates"], args.seed, args.trials)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for index, (unit, impact) in enumerate(zip(design, impacts), start=1):
        speed = scale(float(unit[2]), SPEED_RANGE)
        mass_ratio = scale(float(unit[3]), MASS_RATIO_RANGE)
        theta_deg = scale(float(unit[4]), THETA_RANGE_DEG)
        phi_deg = scale(float(unit[5]), PHI_RANGE_DEG)
        material_index = min(int(unit[6] * len(MATERIALS)), len(MATERIALS) - 1)
        material = MATERIALS[material_index]
        velocity = speed * direction(theta_deg, phi_deg)
        density = BASE_DENSITY * mass_ratio
        mass = BASE_MASS * mass_ratio

        case_name = f"case{index:03d}"
        case_dir = args.output_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        key_path = case_dir / f"{case_name}.key"
        key_lines, ball_center = make_case_key(template, impact, velocity, density, material)
        key_path.write_text("".join(key_lines), encoding="utf-8", newline="\n")
        center = impact["center"]
        rows.append(
            {
                "case": case_name,
                "key_file": str(key_path),
                "panel_pid": PANEL_PID,
                "impact_element_id": impact["eid"],
                "impact_x": f"{center[0]:.8f}",
                "impact_y": f"{center[1]:.8f}",
                "impact_z": f"{center[2]:.8f}",
                "ball_center_x": f"{ball_center[0]:.8f}",
                "ball_center_y": f"{ball_center[1]:.8f}",
                "ball_center_z": f"{ball_center[2]:.8f}",
                "velocity_vx": f"{velocity[0]:.8f}",
                "velocity_vy": f"{velocity[1]:.8f}",
                "velocity_vz": f"{velocity[2]:.8f}",
                "impact_speed": f"{speed:.8f}",
                "mass_ratio": f"{mass_ratio:.8f}",
                "impactor_mass": f"{mass:.10f}",
                "impactor_density": f"{density:.10E}",
                "theta_deg": f"{theta_deg:.8f}",
                "phi_deg": f"{phi_deg:.8f}",
                "material_index": material_index,
                "material_name": material["name"],
                "material_young_mpa": f"{material['young_mpa']:.1f}",
                "material_poisson": f"{material['poisson']:.4f}",
                "min_boundary_distance_mm": f"{impact['min_boundary_distance']:.8f}",
                **{f"lhs_u_{name}": f"{unit[i]:.10f}" for i, name in enumerate(("x", "y", "v", "m", "theta", "phi", "material"))},
            }
        )

    manifest = args.output_root / "case_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["material_name"] for row in rows)
    summary = (
        f"{DESIGN_NAME} constrained LHS design\n"
        f"cases={args.cases}\nseed={args.seed}\ntrials={args.trials}\n"
        f"normalized_maximin_score={score:.8f}\n"
        f"minimum_xy_spacing_mm={min_impact_spacing(impacts):.8f}\n"
        f"speed_range_mm_per_s={SPEED_RANGE}\n"
        f"mass_ratio_range={MASS_RATIO_RANGE}\n"
        f"theta_range_deg={THETA_RANGE_DEG}\nphi_range_deg={PHI_RANGE_DEG}\n"
        f"material_counts={dict(counts)}\n"
        "theta is measured from global +Z; phi is measured in global XY from +X toward +Y.\n"
        "Rigid material class controls E/nu used by contact; mass ratio independently scales density.\n"
    )
    (args.output_root / "lhs_design_summary.txt").write_text(summary, encoding="utf-8", newline="\n")
    print(summary, end="")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
