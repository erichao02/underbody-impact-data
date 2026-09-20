import csv
import math
import random
from pathlib import Path


TEMPLATE = Path("floor_panel_largest_3_pid_2000394_373_floorfrontdriver_with_ball.key")
OUTPUT_ROOT = Path("cases_floorfrontdriver_random_50")

PANEL_PID = 2000394
BALL_PID = 9636
NUM_CASES = 50
RANDOM_SEED = 20260611

MIN_BOUNDARY_DISTANCE_MM = 80.0
TARGET_PAIRWISE_DISTANCE_MM = 120.0
MIN_PAIRWISE_DISTANCE_MM = 60.0
PAIRWISE_RELAX_STEP_MM = 10.0
BALL_GAP_MM = 5.0


def is_keyword(line):
    return line.startswith("*")


def parse_int_fields(line):
    values = []
    for field in line.strip().split():
        try:
            values.append(int(float(field)))
        except ValueError:
            pass
    if len(values) >= 6:
        return values

    fixed = []
    if len(line) >= 48:
        for i in range(6):
            chunk = line[i * 8 : (i + 1) * 8].strip()
            if not chunk:
                fixed.append(0)
                continue
            try:
                fixed.append(int(chunk))
            except ValueError:
                return values
    return fixed if len(fixed) >= len(values) else values


def read_blocks(path):
    block = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if is_keyword(line) and block:
                yield block
                block = [line]
            else:
                block.append(line)
    if block:
        yield block


def parse_node_line(line):
    fields = line.strip().split()
    if len(fields) < 4:
        return None
    try:
        return int(float(fields[0])), (float(fields[1]), float(fields[2]), float(fields[3]))
    except ValueError:
        return None


def format_node(nid, xyz):
    x, y, z = xyz
    return f"{nid:8d}{x:16.8f}{y:16.8f}{z:16.8f}       0       0\n"


def shell_edges(nids):
    clean = []
    for nid in nids:
        if nid > 0 and nid not in clean:
            clean.append(nid)
    if len(clean) < 3:
        return []
    return [tuple(sorted((clean[i], clean[(i + 1) % len(clean)]))) for i in range(len(clean))]


def centroid(points):
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def xy_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def block_is_control_adaptive(block):
    return block[0].strip().upper() == "*CONTROL_ADAPTIVE"


def block_is_end(block):
    return block[0].strip().upper() == "*END"


def parse_template(path):
    blocks = list(read_blocks(path))
    nodes = {}
    shell_elements = []

    for block in blocks:
        key = block[0].strip().upper()
        if key == "*NODE":
            for line in block[1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith("$"):
                    continue
                parsed = parse_node_line(line)
                if parsed:
                    nodes[parsed[0]] = parsed[1]
        elif key == "*ELEMENT_SHELL":
            for line in block[1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith("$"):
                    continue
                fields = parse_int_fields(line)
                if len(fields) >= 6:
                    shell_elements.append(fields[:6])

    panel_elements = [elem for elem in shell_elements if elem[1] == PANEL_PID]
    ball_elements = [elem for elem in shell_elements if elem[1] == BALL_PID]
    if not panel_elements:
        raise SystemExit(f"No panel elements found for PID {PANEL_PID}")
    if not ball_elements:
        raise SystemExit(f"No ball elements found for PID {BALL_PID}")

    ball_node_ids = {nid for elem in ball_elements for nid in elem[2:6] if nid > 0}
    ball_points = [nodes[nid] for nid in ball_node_ids]
    ball_center = centroid(ball_points)
    ball_radius = max(
        max(p[0] for p in ball_points) - min(p[0] for p in ball_points),
        max(p[1] for p in ball_points) - min(p[1] for p in ball_points),
        max(p[2] for p in ball_points) - min(p[2] for p in ball_points),
    ) / 2.0

    edge_counts = {}
    for elem in panel_elements:
        for edge in shell_edges(elem[2:6]):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_node_ids = set()
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_node_ids.update(edge)
    boundary_points = [nodes[nid] for nid in boundary_node_ids if nid in nodes]

    candidates = []
    for elem in panel_elements:
        elem_nodes = [nodes[nid] for nid in elem[2:6] if nid > 0 and nid in nodes]
        if len(elem_nodes) < 3:
            continue
        c = centroid(elem_nodes)
        min_boundary_distance = min(xy_distance(c, bp) for bp in boundary_points)
        if min_boundary_distance >= MIN_BOUNDARY_DISTANCE_MM:
            candidates.append(
                {
                    "eid": elem[0],
                    "center": c,
                    "min_boundary_distance": min_boundary_distance,
                }
            )

    if len(candidates) < NUM_CASES:
        raise SystemExit(
            f"Only {len(candidates)} safe impact candidates found; need {NUM_CASES}. "
            f"Reduce MIN_BOUNDARY_DISTANCE_MM."
        )

    return {
        "blocks": blocks,
        "nodes": nodes,
        "panel_elements": panel_elements,
        "ball_node_ids": ball_node_ids,
        "ball_center": ball_center,
        "ball_radius": ball_radius,
        "candidates": candidates,
        "boundary_node_count": len(boundary_node_ids),
    }


def update_part_adpopt_line(line, adpopt):
    fields = parse_int_fields(line)
    if len(fields) < 3:
        return line
    while len(fields) < 8:
        fields.append(0)
    fields[6] = adpopt
    return "".join(f"{item:10d}" for item in fields[:8]) + "\n"


def update_part_adpopt_block(block, pid, adpopt):
    out = []
    replaced = False
    for line in block:
        if (
            not replaced
            and not line.startswith("*")
            and not line.strip().startswith("$")
            and parse_int_fields(line)[:1] == [pid]
        ):
            out.append(update_part_adpopt_line(line, adpopt))
            replaced = True
        else:
            out.append(line)
    return out


def move_ball_nodes_in_node_block(block, ball_node_ids, old_center, new_center):
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]
    dz = new_center[2] - old_center[2]

    out = []
    for line in block:
        parsed = parse_node_line(line)
        if parsed and parsed[0] in ball_node_ids:
            nid, xyz = parsed
            out.append(format_node(nid, (xyz[0] + dx, xyz[1] + dy, xyz[2] + dz)))
        else:
            out.append(line)
    return out


def make_case_key(template, impact):
    impact_center = impact["center"]
    new_ball_center = (
        impact_center[0],
        impact_center[1],
        impact_center[2] - template["ball_radius"] - BALL_GAP_MM,
    )

    out_blocks = []
    for block in template["blocks"]:
        if block_is_control_adaptive(block):
            continue
        if block_is_end(block):
            continue

        key = block[0].strip().upper()
        if key == "*NODE":
            out_blocks.append(
                move_ball_nodes_in_node_block(
                    block,
                    template["ball_node_ids"],
                    template["ball_center"],
                    new_ball_center,
                )
            )
        elif key == "*PART":
            modified = update_part_adpopt_block(block, PANEL_PID, 0)
            modified = update_part_adpopt_block(modified, BALL_PID, 0)
            out_blocks.append(modified)
        else:
            out_blocks.append(block)

    lines = []
    for block in out_blocks:
        lines.extend(block)
        if block and not block[-1].endswith("\n"):
            lines.append("\n")
    lines.append("*END\n")
    return lines, new_ball_center


def greedy_select(candidates, min_spacing, rng):
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    selected = []

    for candidate in shuffled:
        center = candidate["center"]
        if all(xy_distance(center, item["center"]) >= min_spacing for item in selected):
            selected.append(candidate)
            if len(selected) == NUM_CASES:
                return selected
    return selected


def farthest_point_select(candidates):
    rng = random.Random(RANDOM_SEED)
    first = rng.choice(candidates)
    selected = [first]
    selected_centers = [first["center"]]

    remaining = [candidate for candidate in candidates if candidate is not first]
    while len(selected) < NUM_CASES and remaining:
        best_idx = None
        best_dist = -1.0
        for idx, candidate in enumerate(remaining):
            d = min(xy_distance(candidate["center"], center) for center in selected_centers)
            if d > best_dist:
                best_dist = d
                best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_centers.append(chosen["center"])

    return selected


def select_impacts(candidates):
    spacing = TARGET_PAIRWISE_DISTANCE_MM
    while spacing >= MIN_PAIRWISE_DISTANCE_MM:
        rng = random.Random(RANDOM_SEED)
        selected = greedy_select(candidates, spacing, rng)
        if len(selected) == NUM_CASES:
            return selected, spacing
        spacing -= PAIRWISE_RELAX_STEP_MM

    selected = farthest_point_select(candidates)
    actual_spacing = selected_min_pairwise_distance(selected)
    if actual_spacing < MIN_PAIRWISE_DISTANCE_MM:
        raise SystemExit(
            f"Could not select {NUM_CASES} points with pairwise spacing >= "
            f"{MIN_PAIRWISE_DISTANCE_MM} mm. Best farthest-point spacing: "
            f"{actual_spacing:.6f} mm. Safe candidates: {len(candidates)}"
        )
    return selected, actual_spacing


def selected_min_pairwise_distance(impacts):
    min_dist = float("inf")
    for i, a in enumerate(impacts):
        for b in impacts[i + 1 :]:
            min_dist = min(min_dist, xy_distance(a["center"], b["center"]))
    return min_dist


def nearest_selected_distance(impact, prior_impacts):
    if not prior_impacts:
        return ""
    return min(xy_distance(impact["center"], other["center"]) for other in prior_impacts)


def main():
    if not TEMPLATE.exists():
        raise SystemExit(f"Missing template key: {TEMPLATE}")

    template = parse_template(TEMPLATE)
    impacts, required_spacing = select_impacts(template["candidates"])
    actual_spacing = selected_min_pairwise_distance(impacts)
    OUTPUT_ROOT.mkdir(exist_ok=True)

    manifest_path = OUTPUT_ROOT / "case_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "case",
                "key_file",
                "panel_pid",
                "impact_element_id",
                "impact_x",
                "impact_y",
                "impact_z",
                "ball_center_x",
                "ball_center_y",
                "ball_center_z",
                "min_boundary_distance_mm",
                "nearest_selected_distance_mm",
                "required_pairwise_distance_mm",
            ]
        )

        prior_impacts = []
        for idx, impact in enumerate(impacts, start=1):
            case_name = f"case{idx:03d}"
            case_dir = OUTPUT_ROOT / case_name
            case_dir.mkdir(exist_ok=True)
            key_path = case_dir / f"{case_name}.key"

            key_lines, ball_center = make_case_key(template, impact)
            key_path.write_text("".join(key_lines), encoding="utf-8", newline="\n")

            c = impact["center"]
            nearest = nearest_selected_distance(impact, prior_impacts)
            writer.writerow(
                [
                    case_name,
                    str(key_path),
                    PANEL_PID,
                    impact["eid"],
                    f"{c[0]:.6f}",
                    f"{c[1]:.6f}",
                    f"{c[2]:.6f}",
                    f"{ball_center[0]:.6f}",
                    f"{ball_center[1]:.6f}",
                    f"{ball_center[2]:.6f}",
                    f"{impact['min_boundary_distance']:.6f}",
                    "" if nearest == "" else f"{nearest:.6f}",
                    f"{required_spacing:.6f}",
                ]
            )
            prior_impacts.append(impact)

    print(f"Template: {TEMPLATE}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Generated cases: {NUM_CASES}")
    print(f"Candidate safe points: {len(template['candidates'])}")
    print(f"Boundary nodes avoided: {template['boundary_node_count']}")
    print(f"Minimum boundary distance: {MIN_BOUNDARY_DISTANCE_MM} mm")
    print(f"Required pairwise impact distance used: {required_spacing} mm")
    print(f"Actual minimum selected pairwise distance: {actual_spacing:.6f} mm")
    print(f"Manifest: {manifest_path}")
    print("Adaptive disabled: removed *CONTROL_ADAPTIVE and set ADPOPT=0")
    print("Ball velocity kept from template: VZ = 3464.1")


if __name__ == "__main__":
    main()
