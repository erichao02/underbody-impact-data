# Generation evidence

This directory contains project-authored scripts and design records supporting
the simulation and preprocessing descriptions in the dataset Datasheet.

## Included

- `lhs/`: constrained-LHS generation and staged-extension implementations,
  geometry-specific design summaries, and exact case manifests;
- `impactor/`: construction of the rigid spherical impactor, panel-boundary
  constraints, contact, gravity, termination, and D3PLOT controls;
- `export/`: LS-PrePost command files for nodal displacement and shell effective
  stress export;
- `conversion/`: compact temporal reduction and PyTorch serialization;
- `stress/`: retained LS-PrePost header evidence for the effective-stress
  definition.

## Reproducibility boundary

These files document the algorithms used for the released data. Some generation
scripts require third-party LS-DYNA keyword templates obtained from the cited
upstream model. Those keyword files, LS-DYNA executables, raw D3PLOT databases,
and representative solver logs are not included here. Consequently this
directory is an auditable generation record, not a standalone redistribution
of the upstream finite-element model.

Historical local paths and machine-specific wrapper scripts are intentionally
excluded. Paths appearing in case manifests are relative historical case paths
and contain no user or machine identity.

The repository's code license applies to the project-authored scripts in this
directory. See `../THIRD_PARTY_NOTICES.md` for upstream attribution.
