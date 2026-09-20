from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

import generate_floorfrontdriver_lhs_cases as lhs
import generate_floorfrontdriver_random_cases as base


CASE_ROOT = Path("cases_floorfrontdriver_lhs_100")
START_CASE = 201
END_CASE = 500
SEED = 20260722
TRIALS = 128


def main() -> None:
    manifest_path = CASE_ROOT / "case_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        existing = list(reader)

    expected_existing = START_CASE - 1
    expected_names = [f"case{number:03d}" for number in range(1, START_CASE)]
    if len(existing) != expected_existing or [row["case"] for row in existing] != expected_names:
        raise RuntimeError(
            f"Expected an uninterrupted case001-case{expected_existing:03d} manifest; "
            f"found {len(existing)} rows"
        )
    for number in range(START_CASE, END_CASE + 1):
        case_dir = CASE_ROOT / f"case{number:03d}"
        if case_dir.exists() and any(case_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing case directory: {case_dir}")

    template = base.parse_template(lhs.TEMPLATE)
    used_element_ids = {int(row["impact_element_id"]) for row in existing}
    available = [item for item in template["candidates"] if int(item["eid"]) not in used_element_ids]
    count = END_CASE - START_CASE + 1
    design, impacts, score = lhs.make_design(count, available, SEED, TRIALS)

    added: list[dict[str, object]] = []
    for number, unit, impact in zip(range(START_CASE, END_CASE + 1), design, impacts):
        speed = lhs.scale(float(unit[2]), lhs.SPEED_RANGE)
        mass_ratio = lhs.scale(float(unit[3]), lhs.MASS_RATIO_RANGE)
        theta_deg = lhs.scale(float(unit[4]), lhs.THETA_RANGE_DEG)
        phi_deg = lhs.scale(float(unit[5]), lhs.PHI_RANGE_DEG)
        material_index = min(int(unit[6] * len(lhs.MATERIALS)), len(lhs.MATERIALS) - 1)
        material = lhs.MATERIALS[material_index]
        velocity = speed * lhs.direction(theta_deg, phi_deg)
        density = lhs.BASE_DENSITY * mass_ratio
        mass = lhs.BASE_MASS * mass_ratio

        case_name = f"case{number:03d}"
        case_dir = CASE_ROOT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        key_path = case_dir / f"{case_name}.key"
        key_lines, ball_center = lhs.make_case_key(template, impact, velocity, density, material)
        key_path.write_text("".join(key_lines), encoding="utf-8", newline="\n")
        center = impact["center"]
        row = {
            "case": case_name,
            "key_file": str(key_path),
            "panel_pid": lhs.PANEL_PID,
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
            **{
                f"lhs_u_{name}": f"{unit[index]:.10f}"
                for index, name in enumerate(("x", "y", "v", "m", "theta", "phi", "material"))
            },
        }
        if set(row) != set(fieldnames):
            raise RuntimeError("Existing manifest schema differs from the LHS generator schema")
        added.append(row)

    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(added)
    temporary.replace(manifest_path)

    all_rows = existing + added
    counts = Counter(row["material_name"] for row in all_rows)
    summary = (
        "floorfrontdriver sequential constrained LHS extension\n"
        f"preserved_cases=1-{START_CASE - 1}\n"
        f"added_cases={START_CASE}-{END_CASE}\n"
        f"added_count={count}\ntotal_cases={len(all_rows)}\n"
        f"extension_seed={SEED}\nextension_trials={TRIALS}\n"
        f"extension_normalized_maximin_score={score:.8f}\n"
        f"extension_minimum_xy_spacing_mm={lhs.min_impact_spacing(impacts):.8f}\n"
        f"combined_material_counts={dict(counts)}\n"
        "The added 300 cases form an independent constrained LHS batch.\n"
    )
    (CASE_ROOT / "lhs_extension_201_500_summary.txt").write_text(
        summary, encoding="utf-8", newline="\n"
    )
    print(summary, end="")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
