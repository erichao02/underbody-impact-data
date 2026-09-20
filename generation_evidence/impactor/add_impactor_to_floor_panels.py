import math
import re
from pathlib import Path


REFERENCE = Path("bottomimpact_60JP1.key")
PANEL_GLOB = "floor_panel_largest_*.key"

BALL_SOURCE_PID = 9636
BALL_PID = 9636
BALL_SECID = 9636
BALL_MID = 174
BALL_VZ = 3464.1
TERMINATION_TIME = 0.03

ADAPT_FREQ = 0.0002
ADAPT_TOL = 5.0
ADAPT_OPT = 2
ADAPT_MAX_LEVEL = 2

CONTACT_ID = 57
GRAVITY_CURVE_ID = 251
GRAVITY_Z = 9810.0


def is_keyword(line):
    return line.startswith("*")


def first_int(line):
    fixed = line[:10].strip()
    if fixed:
        try:
            return int(float(fixed))
        except ValueError:
            pass
    fields = line.strip().split()
    if not fields:
        return None
    try:
        return int(float(fields[0]))
    except ValueError:
        return None


def parse_int_fields(line):
    values = []
    for field in line.strip().split():
        try:
            values.append(int(float(field)))
        except ValueError:
            pass
    return values


def parse_fixed_ints(line, width=8, count=6):
    values = []
    for i in range(count):
        chunk = line[i * width : (i + 1) * width].strip()
        if not chunk:
            values.append(0)
            continue
        try:
            values.append(int(chunk))
        except ValueError:
            return None
    return values


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


def data_lines(block):
    for line in block[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("$"):
            yield line


def block_id(block):
    for line in data_lines(block):
        value = first_int(line)
        if value is not None:
            return value
    return None


def parse_part_pid(block):
    for line in data_lines(block):
        fields = parse_int_fields(line)
        if len(fields) >= 3:
            return fields[0]
    return None


def parse_nodes_from_block(block):
    nodes = {}
    for line in block[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        fields = stripped.split()
        try:
            nid = int(float(fields[0]))
            nodes[nid] = (float(fields[1]), float(fields[2]), float(fields[3]))
        except (IndexError, ValueError):
            continue
    return nodes


def parse_shell_line(line):
    fields = parse_int_fields(line)
    if len(fields) >= 6:
        return fields[:6]
    fixed = parse_fixed_ints(line, width=8, count=6)
    if fixed and fixed[0] and fixed[1]:
        return fixed
    return None


def shell_edges(nids):
    clean = []
    for nid in nids:
        if nid > 0 and nid not in clean:
            clean.append(nid)
    if len(clean) < 3:
        return []
    return [
        tuple(sorted((clean[i], clean[(i + 1) % len(clean)])))
        for i in range(len(clean))
    ]


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def center_of_bbox(points):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox(points)
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0)


def extract_reference_ball():
    nodes = {}
    all_shell_elements = []
    ball_part = None
    ball_section = None
    ball_material = None
    control_blocks = []

    for block in read_blocks(REFERENCE):
        key = block[0].strip().upper()
        if key.startswith("*CONTROL_"):
            control_blocks.append(block)
        elif key == "*NODE":
            nodes.update(parse_nodes_from_block(block))
        elif key == "*ELEMENT_SHELL":
            for line in block[1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith("$"):
                    continue
                parsed = parse_shell_line(line)
                if parsed:
                    all_shell_elements.append(parsed)
        elif key == "*PART" and parse_part_pid(block) == BALL_SOURCE_PID:
            ball_part = block
        elif key.startswith("*SECTION_SHELL") and block_id(block) == BALL_SECID:
            ball_section = block
        elif key.startswith("*MAT_RIGID") and block_id(block) == BALL_MID:
            ball_material = block

    ball_elements = [elem for elem in all_shell_elements if elem[1] == BALL_SOURCE_PID]
    ball_node_ids = sorted({nid for elem in ball_elements for nid in elem[2:6] if nid > 0})
    ball_nodes = {nid: nodes[nid] for nid in ball_node_ids}

    if not ball_elements or not ball_nodes:
        raise SystemExit("Failed to extract reference ball mesh")
    if not (ball_part and ball_section and ball_material):
        raise SystemExit("Failed to extract reference ball part/section/material")

    points = list(ball_nodes.values())
    xmin, xmax, ymin, ymax, zmin, zmax = bbox(points)
    center = center_of_bbox(points)
    radius = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2.0
    return {
        "part": ball_part,
        "section": ball_section,
        "material": ball_material,
        "elements": ball_elements,
        "nodes": ball_nodes,
        "center": center,
        "radius": radius,
        "controls": control_blocks,
    }


def parse_panel(path):
    blocks = list(read_blocks(path))
    nodes = {}
    shell_elements = []
    part_ids = []
    boundary_node_sets = {}
    keywords = []

    mode = None
    for block in blocks:
        key = block[0].strip().upper()
        keywords.append(key)
        if key == "*NODE":
            nodes.update(parse_nodes_from_block(block))
        elif key == "*ELEMENT_SHELL":
            for line in block[1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith("$"):
                    continue
                parsed = parse_shell_line(line)
                if parsed:
                    shell_elements.append(parsed)
        elif key == "*PART":
            pid = parse_part_pid(block)
            if pid is not None:
                part_ids.append(pid)

    if len(part_ids) != 1:
        raise SystemExit(f"{path} must contain exactly one panel part, found {part_ids}")
    panel_pid = part_ids[0]
    panel_node_ids = sorted({nid for elem in shell_elements if elem[1] == panel_pid for nid in elem[2:6] if nid > 0})
    panel_points = [nodes[nid] for nid in panel_node_ids]
    xmin, xmax, ymin, ymax, zmin, zmax = bbox(panel_points)

    edge_counts = {}
    for elem in shell_elements:
        if elem[1] != panel_pid:
            continue
        for edge in shell_edges(elem[2:6]):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary_nodes = set()
    for edge, count in edge_counts.items():
        if count == 1:
            boundary_nodes.update(edge)

    return {
        "blocks": blocks,
        "panel_pid": panel_pid,
        "nodes": nodes,
        "shell_elements": shell_elements,
        "bbox": (xmin, xmax, ymin, ymax, zmin, zmax),
        "boundary_nodes": boundary_nodes,
    }


def next_ids(panel):
    used_nodes = set(panel["nodes"])
    used_elems = {elem[0] for elem in panel["shell_elements"]}
    node_base = max(max(used_nodes) + 1, 9000000)
    elem_base = max(max(used_elems) + 1, 9000000)
    return node_base, elem_base


def translated_ball(ball, panel):
    xmin, xmax, ymin, ymax, zmin, zmax = panel["bbox"]
    target_center = (
        (xmin + xmax) / 2.0,
        (ymin + ymax) / 2.0,
        zmin - ball["radius"] - 5.0,
    )
    sx, sy, sz = ball["center"]
    tx, ty, tz = target_center
    return (tx - sx, ty - sy, tz - sz), target_center


def update_part_id_block(block, pid, secid, mid):
    out = []
    replaced = False
    for line in block:
        if not replaced and not line.strip().startswith("$") and not line.startswith("*") and len(parse_int_fields(line)) >= 3:
            out.append(f"{pid:10d}{secid:10d}{mid:10d}         0         0         0         0         0\n")
            replaced = True
        else:
            out.append(line)
    return out


def update_section_id_block(block, secid):
    out = []
    replaced = False
    for line in block:
        if not replaced and not line.strip().startswith("$") and not line.startswith("*") and first_int(line) is not None:
            rest = line[10:] if len(line) > 10 else "\n"
            out.append(f"{secid:10d}{rest}")
            replaced = True
        else:
            out.append(line)
    return out


def update_material_id_block(block, mid):
    out = []
    replaced = False
    for line in block:
        if not replaced and not line.strip().startswith("$") and not line.startswith("*") and first_int(line) is not None:
            rest = line[10:] if len(line) > 10 else "\n"
            out.append(f"{mid:10d}{rest}")
            replaced = True
        else:
            out.append(line)
    return out


def update_control_termination_block(block):
    out = []
    replaced = False
    for line in block:
        if not replaced and not line.strip().startswith("$") and not line.startswith("*"):
            out.append(f"{TERMINATION_TIME:10.4f}         0        0.        0.        0.         0\n")
            replaced = True
        else:
            out.append(line)
    return out


def update_part_adpopt_block(block, adpopt):
    out = []
    replaced = False
    for line in block:
        if not replaced and not line.strip().startswith("$") and not line.startswith("*"):
            fields = parse_int_fields(line)
            if len(fields) >= 3:
                while len(fields) < 8:
                    fields.append(0)
                fields[6] = adpopt
                out.append("".join(f"{item:10d}" for item in fields[:8]) + "\n")
                replaced = True
                continue
        out.append(line)
    return out


def write_control_adaptive(out):
    out.write("*CONTROL_ADAPTIVE\n")
    out.write("$# adpfreq    adptol    adpopt    maxlvl    tbirth    tdeath     lcadp    ioflag\n")
    out.write(
        f"{ADAPT_FREQ:10.6f}{ADAPT_TOL:10.1f}{ADAPT_OPT:10d}{ADAPT_MAX_LEVEL:10d}"
        f"{0.0:10.1f}{TERMINATION_TIME:10.4f}{0:10d}{0:10d}\n"
    )


def write_node_set(out, sid, title, node_ids):
    out.write("*SET_NODE_LIST_TITLE\n")
    out.write(f"{title}\n")
    out.write("$#     sid       da1       da2       da3       da4    solver\n")
    out.write(f"{sid:10d}       0.0       0.0       0.0       0.0MECH\n")
    out.write("$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8\n")
    row = []
    for nid in sorted(node_ids):
        row.append(nid)
        if len(row) == 8:
            out.write("".join(f"{item:10d}" for item in row) + "\n")
            row = []
    if row:
        out.write("".join(f"{item:10d}" for item in row) + "\n")


def format_node(nid, xyz):
    x, y, z = xyz
    return f"{nid:8d}{x:16.8f}{y:16.8f}{z:16.8f}       0       0\n"


def format_shell(eid, pid, nids):
    padded = list(nids[:4])
    while len(padded) < 4:
        padded.append(padded[-1])
    return f"{eid:8d}{pid:8d}{padded[0]:8d}{padded[1]:8d}{padded[2]:8d}{padded[3]:8d}\n"


def strip_end(blocks):
    return [block for block in blocks if block[0].strip().upper() != "*END"]


def output_path(path):
    return path.with_name(path.stem + "_with_ball.key")


def add_ball_to_panel(path, ball):
    panel = parse_panel(path)
    node_base, elem_base = next_ids(panel)
    translation, target_center = translated_ball(ball, panel)
    dx, dy, dz = translation

    node_map = {}
    moved_nodes = {}
    for index, (old_nid, xyz) in enumerate(sorted(ball["nodes"].items())):
        new_nid = node_base + index
        node_map[old_nid] = new_nid
        moved_nodes[new_nid] = (xyz[0] + dx, xyz[1] + dy, xyz[2] + dz)

    moved_elements = []
    for index, elem in enumerate(ball["elements"]):
        new_eid = elem_base + index
        new_nids = [node_map[nid] for nid in elem[2:6] if nid > 0]
        moved_elements.append((new_eid, BALL_PID, new_nids))

    out_path = output_path(path)
    panel_pid = panel["panel_pid"]
    boundary_set_id = 9100000 + (panel_pid % 10000)

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("*KEYWORD\n")
        out.write("$ Control cards copied from bottomimpact_60JP1.key\n")
        for block in ball["controls"]:
            if block[0].strip().upper() == "*CONTROL_TERMINATION":
                out.writelines(update_control_termination_block(block))
            else:
                out.writelines(block)
            if not block[-1].endswith("\n"):
                out.write("\n")
        write_control_adaptive(out)
        out.write("$-------------------------------------------------------------------------------\n")

        for block in strip_end(panel["blocks"]):
            key = block[0].strip().upper()
            if key == "*KEYWORD":
                continue
            if key == "*PART" and parse_part_pid(block) == panel_pid:
                out.writelines(update_part_adpopt_block(block, 1))
                continue
            if key == "*NODE":
                out.writelines(block)
                for nid in sorted(moved_nodes):
                    out.write(format_node(nid, moved_nodes[nid]))
                continue
            if key == "*ELEMENT_SHELL":
                out.writelines(block)
                for eid, pid, nids in moved_elements:
                    out.write(format_shell(eid, pid, nids))
                continue
            if key.startswith("*SET_NODE_LIST"):
                # Regenerate the fixed boundary set after adding the ball.
                continue
            if key.startswith("*BOUNDARY_SPC_SET"):
                continue
            out.writelines(block)

        out.write("$-------------------------------------------------------------------------------\n")
        out.writelines(update_part_id_block(ball["part"], BALL_PID, BALL_SECID, BALL_MID))
        out.writelines(update_section_id_block(ball["section"], BALL_SECID))
        out.writelines(update_material_id_block(ball["material"], BALL_MID))

        out.write("$-------------------------------------------------------------------------------\n")
        write_node_set(out, boundary_set_id, f"PID {panel_pid} outer boundary nodes - fixed", panel["boundary_nodes"])
        out.write("*BOUNDARY_SPC_SET\n")
        out.write("$#    nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz\n")
        out.write(f"{boundary_set_id:10d}         0         1         1         1         1         1         1\n")

        out.write("*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_ID\n")
        out.write(f"{CONTACT_ID:10d}        ball\n")
        out.write("$#    ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr\n")
        out.write(f"{panel_pid:10d}{BALL_PID:10d}         3         3         0         0         0         0\n")
        out.write("$#      fs        fd        dc        vc       vdc    penchk        bt        dt\n")
        out.write("      0.15      0.15       0.0       0.0       0.0         0       0.0       0.0\n")
        out.write("$#     sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf\n")
        out.write("       1.0       1.0       1.0       1.0       1.0       1.0       1.0       1.0\n")
        out.write("$#    soft    sofscl    lcidab    maxpar     sbopt     depth     bsort    frcfrq\n")
        out.write("         1       0.0         0       0.0       0.0         0         0         0\n")
        out.write("$#  penmax    thkopt    shlthk     snlog      isym     i2d3d    sldthk    sldstf\n")
        out.write("       0.0         0         0         0         0         0       0.0       0.0\n")

        out.write("*INITIAL_VELOCITY_RIGID_BODY\n")
        out.write("$      PID|       VX|       VY|       VZ|      VXR|      VYR|      VZR|     ICID|\n")
        out.write(f"{BALL_PID:10d}        0.        0.{BALL_VZ:10.1f}        0.        0.        0.         0\n")

        out.write("*LOAD_BODY_Z\n")
        out.write(f"{GRAVITY_CURVE_ID:10d}{GRAVITY_Z:10.1f}\n")
        out.write("*DEFINE_CURVE\n")
        out.write("$#    lcid      sidr       sfa       sfo      offa      offo    dattyp\n")
        out.write(f"{GRAVITY_CURVE_ID:10d}         0       1.0       1.0       0.0       0.0         0\n")
        out.write("$#                a1                  o1\n")
        out.write("                 0.0                 1.0\n")
        out.write("              1000.0                 1.0\n")

        out.write("*DATABASE_BINARY_D3PLOT\n")
        out.write("$#      dt      lcdt      beam     npltc    psetid\n")
        out.write("  0.000200         0         0         0         0\n")
        out.write("*DATABASE_GLSTAT\n")
        out.write("  0.000100\n")
        out.write("*DATABASE_MATSUM\n")
        out.write("  0.000100\n")
        out.write("*DATABASE_RCFORC\n")
        out.write("  0.000100\n")

        out.write("*END\n")

    print(
        f"{out_path.name}: panel_pid={panel_pid}, ball_nodes={len(moved_nodes)}, "
        f"ball_elements={len(moved_elements)}, ball_center=({target_center[0]:.3f}, "
        f"{target_center[1]:.3f}, {target_center[2]:.3f}), radius={ball['radius']:.3f}"
    )
    return out_path


def main():
    ball = extract_reference_ball()
    print(
        f"Reference ball: nodes={len(ball['nodes'])}, elements={len(ball['elements'])}, "
        f"center=({ball['center'][0]:.3f}, {ball['center'][1]:.3f}, {ball['center'][2]:.3f}), "
        f"radius={ball['radius']:.3f}"
    )
    panel_paths = sorted(Path(".").glob(PANEL_GLOB))
    panel_paths = [path for path in panel_paths if not path.stem.endswith("_with_ball")]
    if len(panel_paths) < 3:
        raise SystemExit(f"Expected at least 3 panel key files matching {PANEL_GLOB}")
    for path in panel_paths[:3]:
        add_ball_to_panel(path, ball)


if __name__ == "__main__":
    main()
