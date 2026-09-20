# Constrained-LHS design

## Design coordinates

Each constrained Latin hypercube design uses seven normalized coordinates:

`[x, y, impact_speed, mass_ratio, theta_deg, phi_deg, material_index]`.

The continuous domains are:

| Parameter | Domain | Unit |
|---|---:|---|
| impact speed | [1732.05, 5196.15] | mm/s |
| mass ratio | [0.75, 1.25] | dimensionless |
| theta | [0, 15] | degree |
| phi | [0, 360] | degree |

Theta is measured from global +Z. Phi is measured in global XY from +X toward
+Y. Speed and angles are converted to the Cartesian velocity vector as

`v = speed * [sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)]`.

Material is a three-level categorical coordinate with balanced counts. It
selects one of the rigid-impactor E/nu pairs listed in
`SIMULATION_PROTOCOL.md`; it is not a continuously interpolated material.

## Geometry-constrained position sampling

Impact positions are panel-shell centroids, not arbitrary points in a
rectangular three-dimensional box. The topological outer boundary is computed
from shell edges occurring in only one element. Candidate centroids must be at
least 80 mm from all outer-boundary nodes in XY.

The first two LHS coordinates are projected to the nearest available candidate
centroid in normalized XY, without reusing a centroid. Consequently:

- X and Y are space-filling design coordinates constrained by the mesh;
- Z is inherited from the selected shell centroid;
- the geometry JSON ranges are observed coordinate extrema, not continuous LHS
  box bounds.

## Maximin selection and seeds

For a requested batch, 128 candidate seven-dimensional LHS designs are
generated. Material coordinates are mapped to category-bin centers for scoring,
and the design maximizing the normalized minimum pairwise distance is retained.

| Geometry | Construction | Seed |
|---|---|---:|
| `floorfrontR` | independent single batch | 20260728 |
| `trunkfloor` | independent single batch | 20260723 |
| `floorfrontdriver`, cases 001--100 | initial batch | 20260721 |
| `floorfrontdriver`, cases 101--200 | complementary nested extension | 20260722 |
| `floorfrontdriver`, cases 201--500 | independent augmentation | 20260722 |

Thus, the final set for each geometry contains 500 constrained-LHS cases, but
the `floorfrontdriver` set is not one monolithic 500-point Latin hypercube.
The three geometries share design bounds but use different eligible position
sets and are not case-wise paired.
