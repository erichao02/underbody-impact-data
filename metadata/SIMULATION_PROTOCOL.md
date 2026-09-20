# Simulation protocol

## Scope

The three panel datasets use the same impactor construction, parameter bounds,
boundary-condition rule, contact definition, solver-output settings, and
compact-result conversion. Their panel meshes, eligible impact locations,
part identifiers, and LHS seeds differ. Equal case identifiers across
geometries are not paired simulations.

## Units

The LS-DYNA models use the tonne--mm--s--N consistent unit system:

| Quantity | Unit |
|---|---|
| coordinate and displacement | mm |
| time | s |
| velocity | mm/s |
| mass | tonne |
| density | tonne/mm^3 |
| force | N |
| stress and Young's modulus | MPa |
| angle | degree |

## Solver

Cases were run with LS-DYNA SMP single precision R12 using the `lsdyna_sp.exe`
distributed through ANSYS v221. The batch configuration used `ncpu=8` and
`memory=400m`. The exact R12 sub-build is not retained for every released case.
The termination time is 0.03 s.

## Rigid spherical impactor

| Setting | Value |
|---|---|
| part ID | 9636 |
| section ID | 9636 |
| material ID | 174 |
| material model | `*MAT_RIGID` |
| element type | shell |
| shell formulation | ELFORM=2 |
| shear factor | SHRF=0.833333 |
| thickness integration | NIP=3 |
| shell thickness | 0.1 mm |
| sphere radius | 12.5 mm |
| initial gap | 5.0 mm |
| reference nominal mass | 0.01 tonne |
| reference density | 5.205e-5 tonne/mm^3 |

For impact target p and unit travel direction v_hat, the initial center is
`c_ball = p - v_hat * (12.5 mm + 5.0 mm)`. No initial angular velocity is
applied. The generator records `impactor_mass = 0.01 tonne * mu` and writes
`impactor_density = 5.205e-5 tonne/mm^3 * mu` into the rigid material. The
released `impactor_mass` should therefore be interpreted as the nominal mass
recorded by the generator.

The material category changes only the rigid impactor's E and nu:

| Index | Label | E (MPa) | nu |
|---:|---|---:|---:|
| 0 | `aluminum_rigid` | 70000 | 0.33 |
| 1 | `titanium_rigid` | 110000 | 0.34 |
| 2 | `steel_rigid` | 210000 | 0.30 |

## Panel boundary and contact

Panel outer-boundary nodes are determined topologically: nodes belonging to a
shell edge used by only one panel element are included in the boundary set.
All six translational and rotational degrees of freedom in that set are fixed
using `*BOUNDARY_SPC_SET`.

The impactor and panel interact through
`*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_ID`. Static and dynamic friction
coefficients are both 0.15. A body acceleration of 9810 mm/s^2 is applied in
global +Z.

Panel materials and shell sections are retained from the cited upstream 2020
Nissan Rogue Version 3 model. Their full keyword cards and the complete vehicle
model are not part of this compact release.

## Outputs

D3PLOT output uses a nominal 0.0002-s interval. Nodal X/Y/Z displacement is
exported with LS-PrePost `ntime 5/6/7`. Shell effective stress is exported with
`etime 9`, labeled `Effective Stress (v-m), ip#max`. See
`TEMPORAL_SAMPLING.md` and the main Datasheet for the compact reduction and
stress semantics.

Representative self-contained keyword inputs and solver `d3hsp`/`matsum`
outputs were retained as audit material but are not distributed in the public
dataset.
