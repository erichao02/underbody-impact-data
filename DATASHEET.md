# Datasheet for the Automotive Underbody Panel Impact Dataset

## Motivation

The dataset supports research on mesh-based surrogate modeling of transient
impact response. It was created to evaluate whether neural operators can map a
finite-element mesh and impact/material conditions to spatially distributed
displacement and shell von Mises effective-stress trajectories.

## Composition

- 3 fixed automotive floor-panel geometries.
- 500 independent LHS cases per geometry; 1,500 cases total.
- 17 aligned states per case.
- Nodal displacement: three Cartesian components.
- Shell-element von Mises effective stress: one scalar per element and state,
  taken as the maximum over all through-thickness integration points.
- Impact position, three-dimensional velocity, mass ratio, and material
  parameters are stored per case.
- Static graph topology and shell element-to-node connectivity are provided per
  geometry.
- One fixed 400/50/50 train/validation/test partition with seed 12345.

Exact tensor shapes are specified in `schema.json` and geometry metadata files.

## Stress definition

The `effective_stress` target is exported from LS-PrePost using `etime 9`,
labeled `Effective Stress (v-m), ip#max`. For each shell element and retained
state, it stores the maximum von Mises equivalent stress across all
through-thickness integration points. The integration-point index producing
the maximum is not retained. Values are reported in MPa.

## Condition definitions and units

The simulations use a tonne--mm--s--N consistent unit system. Coordinates and
displacements are in mm, time is in s, velocity is in mm/s, mass is in tonne,
density is in tonne/mm^3, and stress and Young's modulus are in MPa.

The paper-level inputs map to released fields as follows:

- p is `impact_xyz`, the centroid of a selected eligible panel shell;
- v is `velocity_xyz`, the rigid impactor's initial translational velocity;
- mu is `mass_ratio`, a dimensionless scale factor in [0.75, 1.25];
- E is `material_young_mpa`, the rigid-impactor Young's modulus;
- nu is `material_poisson`, the rigid-impactor Poisson ratio.

The impact speed is sampled in [1732.05, 5196.15] mm/s. Theta is sampled in
[0, 15] degrees from global +Z, and phi is sampled in [0, 360] degrees in
global XY from +X toward +Y. Cartesian velocity is computed from speed and
these two angles. The three material categories are discrete rigid-impactor
E/nu pairs: (70000 MPa, 0.33), (110000 MPa, 0.34), and
(210000 MPa, 0.30).

Mass ratio scales the generator's reference impactor mass and density:
`impactor_mass = 0.01 tonne * mu` and
`impactor_density = 5.205e-5 tonne/mm^3 * mu`. The mass field is the nominal
mass recorded by the generator.

## Collection and simulation process

Conditions were generated with a seven-dimensional constrained Latin hypercube
over position X/Y, speed, mass ratio, theta, phi, and material class. Eligible
impact positions are panel-shell centroids at least 80 mm from the topological
outer boundary. Normalized LHS position coordinates are mapped to unused
eligible centroids, so Z is inherited from the selected shell and is not an
independent continuous coordinate. Among 128 trial designs, the normalized
maximin design is retained. `metadata/LHS_DESIGN.md` records the exact
geometry-specific batching and seeds.

The simulations were executed with LS-DYNA SMP single precision R12 through
ANSYS v221 `lsdyna_sp.exe`, using `ncpu=8` and `memory=400m`, on panel geometry
derived from the 2020 Nissan Rogue finite-element model Version 3. The rigid
spherical-shell impactor has radius 12.5 mm, thickness 0.1 mm, ELFORM 2,
SHRF 0.833333, NIP 3, and a 5.0-mm initial gap. Panel outer-boundary nodes are
fixed in all six degrees of freedom. Impactor--panel interaction uses automatic
surface-to-surface contact with static and dynamic friction coefficients of
0.15. A body acceleration of 9810 mm/s^2 acts in global +Z. Further details are
given in `metadata/SIMULATION_PROTOCOL.md`.

Raw solver databases and curve text are not included. Panel material and shell
definitions remain those of the upstream Version 3 model. The exact LS-DYNA
R12 sub-build is not retained for every released case.

## Preprocessing

The released case tensors are the compact 17-state inputs to downstream data
preparation. They have not been reduced to a single peak state. The accompanying
`build_peak_targets.py` derives the paper task by selecting the state containing
the largest valid nodal displacement magnitude and using stress from the same
state.

The solver requests D3PLOT output every 0.0002 s through 0.03 s. Compact
conversion retains every tenth raw state and appends the final state. Both
nodal and element fields use indices `[0, 10, 20, ..., 150, 151]` in every
released case. The first 16 retained states have a nominal 0.002-s spacing;
the appended terminal state can be much closer to index 150. Exact times are
stored per case.

Peak time `t*` is an argmax over valid nodes and these 17 retained states only.
It is therefore a discrete, temporally quantized label rather than a
continuous-time solver maximum. See `metadata/TEMPORAL_SAMPLING.md`.

Some source nodes may require filled values; each case retains
`raw_valid_node_mask`, `filled_node_mask`, `filled_node_count`, and
`valid_node_mask` to make that processing explicit.

## Data quality

The release validator checks:

- exactly 500 cases per geometry;
- geometry-specific displacement and stress shapes;
- 17 states in each field;
- finite displacement and stress values;
- monotonic displacement and element time arrays;
- exact alignment of displacement and stress time arrays;
- valid static graph and shell-element connectivity;
- complete and disjoint split coverage;
- complete case-file membership and SHA-256 case digests.

The `floorfrontR` revision additionally requires the source stress quality
audit described in `metadata/floorfrontR_DATA_NOTE.md`.

## Recommended uses

- full-field transient surrogate modeling;
- graph neural operators and mesh-based learning;
- peak-event displacement/stress prediction;
- temporal interpolation or sequence modeling within the released protocol;
- controlled comparisons on fixed meshes;
- simulation-based screening research.

## Out-of-scope or unsupported uses

- safety certification or replacement of final CAE/physical testing;
- claims of arbitrary-geometry generalization;
- treating same-numbered cases across geometries as physical pairs;
- claims about real-world crash response without external validation;
- mixing earlier internal `floorfrontR` artifacts with this release;
- interpreting the public test labels as a permanently hidden benchmark.

## Splits and benchmark integrity

The full v1.1 release includes labels for train, validation, and test cases.
Consequently, the test split reproduces the paper protocol but is not a hidden
benchmark after publication. New benchmark work should define a separate
private evaluation set or use an evaluation server.

## Personal and sensitive information

The data contain no human participants, personal data, or user-generated
content. The main reuse consideration is the documented provenance of the
underlying vehicle mesh.

## Distribution and maintenance

The archival host is the Hugging Face Hub, with a version tag. Changes to data
files require a new dataset version. Metadata changes should be documented
without silently replacing data.

## Licensing

Project-authored code, documentation, metadata, and derived numerical results
are released under the MIT License. The panel meshes are derived from the cited
CCSA/NHTSA vehicle model; upstream attribution is preserved and the upstream
model is not represented as MIT-licensed project-authored content. Provenance
and attribution are documented in `THIRD_PARTY_NOTICES.md`.
