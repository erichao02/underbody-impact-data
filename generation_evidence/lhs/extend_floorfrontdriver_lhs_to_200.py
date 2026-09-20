from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

import generate_floorfrontdriver_lhs_cases as lhsgen
import generate_floorfrontdriver_random_cases as base


ROOT = Path("cases_floorfrontdriver_lhs_100")
TOTAL_CASES = 200
SEED = 20260722
TRIALS = 128
PARAMETER_NAMES = ("v", "m", "theta", "phi", "material")


def complementary_values(rows: list[dict], name: str, rng: np.random.Generator) -> np.ndarray:
    occupied = {int(np.floor(TOTAL_CASES * float(row[f"lhs_u_{name}"]))) for row in rows}
    missing = np.asarray(sorted(set(range(TOTAL_CASES)) - occupied), dtype=np.int64)
    if len(missing) != 100:
        raise RuntimeError(f"{name}: expected 100 unoccupied nested-LHS bins, found {len(missing)}")
    rng.shuffle(missing)
    return (missing + rng.random(len(missing))) / TOTAL_CASES


def select_design(template: dict, old_rows: list[dict]) -> tuple[np.ndarray, list[dict], float]:
    old_eids = {int(row["impact_element_id"]) for row in old_rows}
    candidates = [item for item in template["candidates"] if int(item["eid"]) not in old_eids]
    old_design = np.asarray(
        [
            [float(row[f"lhs_u_{name}"]) for name in ("x", "y", *PARAMETER_NAMES)]
            for row in old_rows
        ],
        dtype=np.float64,
    )
    root_rng = np.random.default_rng(SEED)
    best_design = None
    best_impacts = None
    best_score = -np.inf

    for _ in range(TRIALS):
        rng = np.random.default_rng(int(root_rng.integers(0, np.iinfo(np.int64).max)))
        design = np.empty((100, 7), dtype=np.float64)
        design[:, :2] = lhsgen.lhs(100, 2, rng)
        for column, name in enumerate(PARAMETER_NAMES, start=2):
            design[:, column] = complementary_values(old_rows, name, rng)
        impacts, mapped = lhsgen.map_xy_to_panel(design, candidates, rng)
        scored_new = mapped.copy()
        scored_old = old_design.copy()
        scored_new[:, 6] = (np.floor(scored_new[:, 6] * 3) + 0.5) / 3
        scored_old[:, 6] = (np.floor(scored_old[:, 6] * 3) + 0.5) / 3
        score = lhsgen.pairwise_min_distance(np.vstack((scored_old, scored_new)))
        if score > best_score:
            best_design, best_impacts, best_score = mapped, impacts, score

    assert best_design is not None and best_impacts is not None
    return best_design, best_impacts, best_score


def main() -> None:
    manifest = ROOT / "case_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    if len(old_rows) == TOTAL_CASES:
        print("[OK] manifest already contains 200 cases; nothing to generate")
        return
    if len(old_rows) != 100:
        raise RuntimeError(f"Expected existing 100-case manifest, found {len(old_rows)} rows")

    template = base.parse_template(lhsgen.TEMPLATE)
    design, impacts, score = select_design(template, old_rows)
    new_rows: list[dict] = []
    for case_number, (unit, impact) in enumerate(zip(design, impacts), start=101):
        speed = lhsgen.scale(float(unit[2]), lhsgen.SPEED_RANGE)
        mass_ratio = lhsgen.scale(float(unit[3]), lhsgen.MASS_RATIO_RANGE)
        theta_deg = lhsgen.scale(float(unit[4]), lhsgen.THETA_RANGE_DEG)
        phi_deg = lhsgen.scale(float(unit[5]), lhsgen.PHI_RANGE_DEG)
        material_index = min(int(unit[6] * len(lhsgen.MATERIALS)), len(lhsgen.MATERIALS) - 1)
        material = lhsgen.MATERIALS[material_index]
        velocity = speed * lhsgen.direction(theta_deg, phi_deg)
        density = lhsgen.BASE_DENSITY * mass_ratio
        mass = lhsgen.BASE_MASS * mass_ratio
        case_name = f"case{case_number:03d}"
        case_dir = ROOT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        key_path = case_dir / f"{case_name}.key"
        key_lines, ball_center = lhsgen.make_case_key(template, impact, velocity, density, material)
        key_path.write_text("".join(key_lines), encoding="utf-8", newline="\n")
        center = impact["center"]
        new_rows.append(
            {
                "case": case_name,
                "key_file": str(key_path),
                "panel_pid": lhsgen.PANEL_PID,
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
        )

    all_rows = old_rows + new_rows
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    temporary.replace(manifest)

    counts = Counter(row["material_name"] for row in all_rows)
    summary = (
        f"cases={len(all_rows)}\nseed_extension={SEED}\ntrials={TRIALS}\n"
        f"combined_normalized_maximin_score={score:.8f}\n"
        f"material_counts={dict(counts)}\n"
        "case001-case100 preserved; case101-case200 use complementary 200-bin nested LHS strata.\n"
    )
    (ROOT / "lhs_design_summary_200.txt").write_text(summary, encoding="utf-8", newline="\n")
    print(summary, end="")


if __name__ == "__main__":
    main()
