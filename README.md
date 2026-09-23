# Slocum-AD2CP

Slocum-AD2CP processes Nortek AD2CP (Acoustic Doppler Current Profiler) data
from Teledyne Webb Research Slocum gliders into depth-resolved ocean current
profiles. A glider-mounted ADCP only measures velocity *relative to the
glider*; this package combines those raw beam measurements with the
glider's own GPS-derived dive-averaged current to solve for the actual
ocean current at each depth, dive by dive.

Please cite this package using this [DOI](https://doi.org/10.5281/zenodo.7416126).
The intention for publishing this code on GitHub is twofold: 1) to make
glider based acoustic current profiler data easier to work with and 2) to
improve upon the processing workflow and make it more transparent. For these
reasons, I encourage comments, questions, concerns, and for users of this
code to report any potential bugs. Please do not hesitate to reach out to me
at jgradone@marine.rutgers.edu.

<img width="680" alt="Screen Shot 2022-11-30 at 1 01 27 PM" src="https://user-images.githubusercontent.com/43152605/204873998-595184d4-4221-49bf-9134-cc85f56b9bb0.png">

## Contents

- [Installation](#installation)
- [Processing raw AD2CP data](#processing-raw-ad2cp-data)
- [Quickstart](#quickstart)
- [The processing pipeline](#the-processing-pipeline)
- [Configuration flags](#configuration-flags)
- [Known issues](#known-issues)
- [Changelog](#changelog)

## Installation

```
pip install slocum_ad2cp
```

The AD2CP processing itself only needs `numpy`, `xarray`, `pandas`,
`netCDF4`, `scipy`, `gsw`, and `dask` (installed automatically). Getting
glider flight data pulls in one of two optional backends, matching the
`extras_require` groups in `setup.py`:

```
pip install "slocum_ad2cp[erddap]"   # ERDDAP tabledap access, via erddapy
pip install "slocum_ad2cp[dbd]"      # raw .dbd/.ebd card dumps, via dbdreader
pip install "slocum_ad2cp[all]"      # both
```

## Processing raw AD2CP data

This package assumes you've already converted your AD2CP's raw binary
output to NetCDF using Nortek's MIDAS software, producing one or more
`.ad2cp.*.nc` files per deployment.

## Quickstart

This is the shape of a full processing run for one deployment — load glider
flight data, load AD2CP data, calibrate the compass once for the whole
deployment, then run the per-segment pipeline. See
[docs/PIPELINE.md](docs/PIPELINE.md) for what each step does physically, and
`notebooks/02_Slocum_AD2CP_Processing_Example.ipynb` for a complete, runnable
version of this against real data.

```python
import numpy as np
import slocum_ad2cp

# 1. Glider flight data (dive-averaged current, depth, heading), segmented
#    into one row per dive/climb. Swap source="dbd" to read raw card dumps
#    instead, before a deployment reaches ERDDAP.
gdf, segments = slocum_ad2cp.load_glider(
    source="erddap",
    ds_id="ru29-20250715T1838-trajectory-raw-delayed",
    server="http://slocum-data.marine.rutgers.edu/erddap",
)

# 2. Raw AD2CP beam data for the whole deployment.
ad2cp = slocum_ad2cp.load_ad2cp("RU29_Barbados.ad2cp.00000.nc", mean_lat=11)

# 3. Compass calibration, once for the whole deployment (needs many
#    attitude combinations to fit against).
ad2cp = slocum_ad2cp.correct_ad2cp_heading(ad2cp)

# 4. Per-segment pipeline.
profiles = []
for _, seg in segments.iterrows():
    subset = ad2cp.sel(time=slice(seg.start_time, seg.end_time))
    if subset.sizes["time"] == 0:
        continue

    subset = slocum_ad2cp.mag_var_correction_ad2cp_ds(
        subset, heading_var="CorrectedHeading", mag_var_arr=seg.mag_var_deg
    )
    subset = slocum_ad2cp.correct_sound_speed(subset)
    subset = slocum_ad2cp.qaqc_pre_coord_transform(subset, corr_threshold=50, max_amplitude=75)
    subset = slocum_ad2cp.beam_true_depth(subset)
    subset = slocum_ad2cp.binmap_adcp(subset)
    subset = slocum_ad2cp.calcAHRS(subset)
    subset = slocum_ad2cp.beam2enu(subset)
    subset = slocum_ad2cp.qaqc_post_coord_transform(
        subset, high_velocity_threshold=1.5, surface_depth_to_filter=5
    )

    # segments.u_dac / v_dac are already magnetic-variation-corrected
    # (segment_table_from_gdf did this once for the whole table).
    O_ls, G_ls, bin_new, obs_per_bin = slocum_ad2cp.inversion(
        subset.UVelocity.values, subset.VVelocity.values,
        dz=10, u_daverage=seg.u_dac, v_daverage=seg.v_dac,
        bins=subset["VelocityRange"].values, depth=subset["Depth"].values,
        wDAC=5, wSmoothness=1,
    )
    profiles.append({"u": np.real(O_ls), "v": np.imag(O_ls), "depth": bin_new})
```

## The processing pipeline

Full walkthrough, step by step, with the physical meaning of each stage:
**[docs/PIPELINE.md](docs/PIPELINE.md)**.

In brief: glider flight data and AD2CP beam data are loaded independently
and matched up per dive/climb segment. Each segment then runs through
magnetic variation correction, sound-speed correction, pre-transform QC,
beam-depth geometry, bin mapping, the AHRS rotation, the beam→ENU coordinate
transform, post-transform QC, and finally the least-squares shear inversion
(`inversion`) that produces the absolute velocity profile for that dive.

## Configuration flags

- **`honour_pitch_selection`** (`beam2enu`, default `True`) — use the
  pitch-dependent beam selection correctly. Versions up to 2.0.0 effectively
  ignored it; see [docs/PIPELINE.md](docs/PIPELINE.md#configuration-flags)
  for the real-world impact this had.
- **`use_loop`** (`beam_true_depth`, `calcAHRS`, `beam2enu`, `inversion`,
  default `False`) — run the original per-ping Python loop instead of the
  vectorized implementation, for cross-checking. Both are verified
  numerically equivalent; the loop is much slower.

## Known issues

- `shear_method` calls three helper functions that don't exist anywhere in
  this package and will raise `NameError` if called. Use `inversion()`.
- See [docs/PIPELINE.md](docs/PIPELINE.md#known-issues) for details and any
  further issues found since.

## Changelog

**Unreleased, 2026-09-22**

- Wrote up the processing pipeline end-to-end in [docs/PIPELINE.md](docs/PIPELINE.md), expanded this README into a real installation/quickstart guide, and added docstrings to the functions in `make_dataset.py` and `analysis.py` that were previously undocumented (the `check_*_beam_range*` helpers, `binmap_adcp`, `cell_vert`, `correct_sound_speed`, `qaqc_pre_coord_transform`, `qaqc_post_coord_transform`, `inversion`, `mag_var_correction`, `shear_method`, `ellipsoid_fit`, `dist_from_lat_lon`, `gsw_rho`). In the process, found and documented two pre-existing issues: `shear_method` calls undefined helper functions and cannot currently run, and `load_ad2cp`'s docstring claimed a `(ds, group)` return that the implementation never provided (docstring corrected to match the actual single-value return).
- Vectorized the per-ping Python loops in `make_dataset.py` (`beam_true_depth`, `calcAHRS`, `beam2enu`, and the `inversion()` G-matrix construction) into NumPy/SciPy array operations, for performance on large deployments. No algorithmic change: verified numerically equivalent to the loop-based version on real RU29 Barbados 2025 data (u/v correlation > 0.9999998, with the residual fully explained by `scipy.sparse.linalg.lsqr`'s iterative floating-point convergence order rather than any change in the computation). The vectorized `beam2enu` implements `honour_pitch_selection` via a per-timestep transform-matrix selection and was checked against the fixed loop-based version below on both real data and synthetic edge cases.
- Added a `use_loop` parameter (default `False`) to `beam_true_depth`, `calcAHRS`, `beam2enu`, and `inversion()`, so the original per-ping Python loop can be selected instead of the vectorized path at call time, for cross-checking. Both paths are verified to agree bit-for-bit on real RU29 Barbados 2025 data.

**Unreleased, 2026-09-16**

- Added a `glider.py` module with an interchangeable glider data backend: alongside the existing `get_erddap_dataset` path, `load_glider_dbd`/`load_glider`(`source="dbd"`) reads raw Slocum dbd/ebd files (and their LZ4-compressed dcd/ecd counterparts, needs `dbdreader>=0.6`) directly, useful before a deployment is published to ERDDAP. Both backends return the same segment-table schema.
- Added `correct_ad2cp_mounting` / `detect_roll_offset` to `make_dataset.py`. Some Slocum payload-bay AD2CP mounts report roll 180 degrees from the glider's own frame; left uncorrected, `cell_vert` returns negative cell depths and `binmap_adcp` silently drops every bin. Opt-in, not called by the existing pipeline.
- **Fixed a bug in `beam2enu`** (`make_dataset.py`): the function computed a pitch-dependent 3-of-4-beam selection (dropping the aft beam on downcasts, the forward beam on upcasts, per the glider's actual attitude) and then unconditionally discarded that choice, always applying the beam 2/3/4 transform regardless of pitch. `beam2enu` now takes `honour_pitch_selection` (default `True`, the corrected behaviour).
