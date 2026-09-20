# floorfrontR data-quality note

The only valid floorfrontR revision in this release is:

```text
review_release_v1
```

Fifty-eight source cases in an earlier internal build exhibited degenerate
von Mises effective-stress frames. Those cases were rebuilt from the solver
outputs, after which all 500 compact 17-state cases were audited.

Release requirements:

- all 500 source and released cases must be finite and match the fixed mesh;
- displacement and von Mises effective-stress tensors must each contain 17 aligned states;
- every non-initial source stress frame must satisfy dominant-value fraction
  at most `0.01` and unique-value fraction at least `0.9`;
- earlier internal `floorfrontR` cases, normalization, checkpoints, and metrics
  are obsolete and are not part of this release.

The internal final source audit contained no invalid case rows. The public
release validator independently checks the released compact tensors but cannot
recreate a solver-source semantic audit without the omitted solver databases.
