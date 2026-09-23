# Processing pipeline

This document walks through how raw Nortek AD2CP + Slocum glider data becomes
a depth-resolved ocean current profile, in the order the functions actually
run. For installation and a quickstart, see the main [README](../README.md).
For what each function's arguments mean in detail, see its docstring in
`src/slocum_ad2cp/`.

## The two inputs, and why they're separate

A glider-mounted ADCP measures velocity *relative to the glider*. To turn
that into an absolute ocean current, this package needs two independent data
streams for the same dive:

1. **AD2CP beam data** — raw along-beam Doppler velocities, correlation,
   amplitude, and the instrument's own attitude (heading/pitch/roll),
   sampled every ping. Lives in the `.ad2cp.*.nc` files produced by Nortek's
   MIDAS software.
2. **Glider flight data** — the glider's own GPS-derived dive-averaged
   current (DAC), depth, and heading, sampled at a much coarser rate (once
   per dive/climb cycle for DAC, more often for depth/heading). This is
   what tells the inversion what the *glider's* velocity through water was,
   which is otherwise indistinguishable from ocean current in the raw beam
   data.

These are loaded independently, matched up by segment (one dive or climb),
and only combined at the very last step (`inversion`).

## Getting glider flight data

Two backends, both returning the same shape of data:

- **`analysis.get_erddap_dataset(ds_id, server, variables, filetype)`** — the
  original, low-level path. Pulls a raw ERDDAP tabledap query and returns it
  as a dataframe or xarray Dataset, one row per glider status update. This
  is what the example notebooks and the reference processing scripts use;
  segmenting by dive/climb (grouping on `source_file`) is left to the
  caller.
- **`glider.load_glider(source="erddap" | "dbd", **kwargs)`** — a newer,
  higher-level path that also reads raw `.dbd`/`.ebd` card dumps directly
  (via `dbdreader`, install with the `dbd` extra), useful before a
  deployment has been published to ERDDAP. Returns `(gdf, segments)`:
  `gdf` is the same kind of flat per-record dataframe as above, and
  `segments` is a one-row-per-dive summary table (`glider.SEGMENT_COLUMNS`:
  start/end time, start/end/mid lat-lon, DAC in a few forms, magnetic
  variation, max depth, record count) already computed — the scalar
  extraction every per-segment loop otherwise has to do inline.

Either way, a "segment" is one dive or climb, identified by ERDDAP's
`source_file` (or the dbd file's basename).

## Getting AD2CP data

`load_ad2cp(ncfile, mean_lat)` opens one or more `.ad2cp.*.nc` files (via
`xarray.open_mfdataset` when given a list), tries the `Data/Average/` group
first and falls back to `Data/Burst/`, and converts the instrument's
`Pressure` to `Depth` via `gsw.z_from_p` at the given latitude. The result is
one dataset covering the whole deployment, which the caller then slices per
segment using the glider flight data's start/end times for that dive.

Before slicing, run the compass calibration once on the *whole* deployment
(it needs many attitude/heading combinations to fit against, not just one
dive's worth):

- **`correct_ad2cp_heading(ds)`** — soft-iron magnetometer calibration.
  Bins the deployment by pitch, fits an ellipsoid to the raw magnetometer
  readings in each bin (`ellipsoid_fit`, since nearby ferrous material
  distorts what should be a sphere centered on the origin), recenters, and
  recomputes heading (`calc_heading`/`calc_tilt_matrix`) from the corrected
  field.
- If your AD2CP reports roll in a frame rolled 180° from the glider's own
  (mount-dependent, and *not* auto-detected by default) — `detect_roll_offset`
  / `correct_ad2cp_mounting` fix that up front. Left uncorrected, `cell_vert`
  silently returns negative cell depths and `binmap_adcp` drops every bin.

## Per-segment pipeline

For each dive/climb segment, in order:

| # | Function | What it does physically |
|---|----------|--------------------------|
| 1 | `mag_var_correction` | Rotates the glider's DAC vector from magnetic to true north, using the segment's mean magnetic variation. **Already applied** if you got your segment table from `glider.load_glider()`/`segment_table_from_gdf` — `segments.u_dac`/`v_dac` are pre-corrected (see `u_dac_raw`/`v_dac_raw` for the uncorrected values). Only a manual step if you built your own segment loop against `get_erddap_dataset` directly. |
| 2 | `mag_var_correction_ad2cp_ds` | Same rotation, applied to the AD2CP's own corrected heading. |
| 3 | `correct_sound_speed` | Rescales beam velocities by (measured sound speed) / 1500 m/s — the AD2CP assumes 1500 m/s internally when converting Doppler shift to velocity. |
| 4 | `qaqc_pre_coord_transform` | Drops beam velocities with low correlation or saturated amplitude — noisy or off-target returns — while everything is still in beam coordinates. |
| 5 | `beam_true_depth` | Computes each beam's *true* cell depth given the glider's pitch/roll at that instant (a cell nominally 2 m along a tilted beam isn't 2 m straight down). |
| 6 | `binmap_adcp` | Interpolates each beam's velocities from their true depths onto the instrument's regular depth grid, so beams can be compared cell-by-cell. |
| 7 | `calcAHRS` | Builds the 3×3 rotation matrix (heading × tilt) that will carry instrument-frame velocities into earth-frame ENU. |
| 8 | `beam2enu` | Selects the 3-of-4 beams appropriate to the glider's pitch (downcast vs. upcast), transforms beam → XYZ → ENU. See **Configuration flags** below — this is the step with a documented correctness fix. |
| 9 | `qaqc_post_coord_transform` | Drops physically implausible velocities (above `high_velocity_threshold`), the bin immediately below the glider (contaminated by its own wake), and any ping where the glider itself was shallower than `surface_depth_to_filter`. |
| 10 | `inversion` | Solves the per-segment least-squares shear inversion (Todd et al. 2017 / Visbeck 2002): one unknown glider velocity per ping plus one unknown ocean velocity per depth bin, constrained to match the segment's DAC, producing the final absolute velocity profile for this dive. |

Steps 5-9 operate on the whole `xarray.Dataset` and return it with new
variables assigned; step 10 takes plain numpy arrays out of that dataset and
returns the profile as plain arrays (`O_ls`, `G_ls`, `bin_new`,
`obs_per_bin`) — see its docstring for exactly what each represents. The
calling script is then responsible for stacking every segment's `O_ls`
profile into a combined depth × time grid.

`shear_method` is an alternative to `inversion()` (shear-integration rather
than simultaneous least-squares) but is **not currently usable** — see
[Known issues](#known-issues).

## Configuration flags

Two independent boolean knobs, threaded through several of the functions
above:

- **`honour_pitch_selection`** (`beam2enu`, default `True`) — whether the
  pitch-dependent choice of which 3 beams to use (dropping the aft beam on
  a downcast, the forward beam on an upcast) actually gets applied to the
  beam→XYZ transform matrix, or discarded in favor of always using the
  upcast matrix. Versions of this package up to 2.0.0 effectively always
  passed `False` (the choice was computed and then silently thrown away) —
  on a real deployment (ru37, Cayman 2026) that produced a horizontal
  velocity direction incoherent with the glider's own compass (median
  absolute deviation 117° vs. an expected ~12°) and biased the mean
  vertical velocity to -0.14 m/s instead of ~0.00 m/s. Leave this at the
  default unless you specifically need to reproduce old output.
- **`use_loop`** (`beam_true_depth`, `calcAHRS`, `beam2enu`, `inversion`,
  default `False`) — selects the original per-ping Python loop
  implementation instead of the vectorized NumPy/SciPy implementation. Both
  are verified numerically equivalent (see the repo's Changelog); the loop
  is much slower and is only useful for cross-checking the vectorized path
  against the original algorithm.

## Known issues

- **`shear_method` is broken.** It calls `calc_ensemble_shear`, `bin_attr`,
  and `shear_to_vel`, none of which are defined anywhere in this package —
  calling it raises `NameError`. It's still exported in `__all__`. Use
  `inversion()` instead.
- **`load_ad2cp` returns one value, not two.** Its docstring previously
  claimed `(ds, group)`; the implementation only ever returned `ds`. Fixed
  in the docstring — if you have code written against the old docstring
  expecting a `(ds, group)` unpack, drop the second value.
