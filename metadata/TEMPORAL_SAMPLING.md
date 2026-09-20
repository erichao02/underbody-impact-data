# Temporal sampling and peak-event time

## Raw solver output

Each LS-DYNA simulation terminates at 0.03 s. D3PLOT is requested at a nominal
0.0002-s interval. Actual floating-point state times can differ slightly from
the nominal values and are therefore stored explicitly.

## Compact 17-state sequence

Compact conversion uses stride 10 on both nodal displacement and shell stress,
then appends the final available state when it is not already selected. Across
all released cases, both `time_indices` and `element_time_indices` equal:

`[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 151]`.

The first 16 retained states therefore have a nominal spacing of 0.002 s. Index
151 is the appended terminal state and can be separated from index 150 by much
less than 0.002 s. Consumers must use the stored `time` and `element_time`
arrays rather than reconstructing timestamps from an assumed uniform interval.

## Discrete peak-event definition

Let T17 be the 17 retained states, Vvalid the valid-node set, and u_i(t) the
three-dimensional nodal displacement. The supplied peak-target builder uses

`t* = argmax_(t in T17) max_(i in Vvalid) ||u_i(t)||_2`.

The implementation flattens the node-by-retained-state magnitude array and
uses the first maximum returned by `torch.argmax`. The displacement field and
the shell von Mises effective-stress field at the same selected retained state
form the paired peak-event target.

Accordingly, `t*` is quantized to the released 17-state grid. It is not an
interpolated time and is not guaranteed to equal the continuous-time maximum
over every raw solver state.
