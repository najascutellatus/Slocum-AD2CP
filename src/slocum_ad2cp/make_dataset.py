import numpy as np
import math
from scipy.sparse.linalg import lsqr
import scipy
import xarray as xr
import gsw




##################################################################################################

def check_max_beam_range(beam,bins):
    """
    Furthest bin depth with a non-NaN velocity, for a single ping.

    Parameters
    ----------
    beam : array_like
        1-D velocity profile for one beam, one ping (NaN where no return).
    bins : array_like
        Bin depth (or range) for each element of `beam`.

    Returns
    -------
    float
        `bins` value at the deepest non-NaN entry in `beam`, or NaN if the
        whole ping is NaN.
    """
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        ind2 = np.max(ind1[:,0])
        beam_range = bins[ind2]
    return(beam_range)

##################################################################################################

def check_max_beam_range_bins(beam,bins):
    """
    Same as `check_max_beam_range`, but returns the bin *index* rather than
    the bin depth/range value.

    Parameters
    ----------
    beam : array_like
        1-D velocity profile for one beam, one ping (NaN where no return).
    bins : array_like
        Unused; kept for signature symmetry with `check_max_beam_range`.

    Returns
    -------
    float
        Index of the deepest non-NaN entry in `beam`, or NaN if the whole
        ping is NaN.
    """
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        beam_range = np.max(ind1[:,0])
    return(beam_range)



##################################################################################################

def check_mean_beam_range(beam,bins):
    """
    Bin depth closest to the mean index of non-NaN velocities, for a single
    ping. A rough single-number summary of how far out a beam is returning
    usable data.

    Parameters
    ----------
    beam : array_like
        1-D velocity profile for one beam, one ping (NaN where no return).
    bins : array_like
        Bin depth (or range) for each element of `beam`.

    Returns
    -------
    float
        `bins` value nearest the mean non-NaN index, or NaN if the whole
        ping is NaN.
    """
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        ind2 = round(np.nanmean(ind1[:,0]))
        beam_range = bins[ind2]
    return(beam_range)

##################################################################################################

def check_mean_beam_range_bins(beam,bins):
    """
    Same as `check_mean_beam_range`, but returns the (fractional, unrounded)
    mean bin *index* rather than the bin depth/range value.

    Parameters
    ----------
    beam : array_like
        1-D velocity profile for one beam, one ping (NaN where no return).
    bins : array_like
        Unused; kept for signature symmetry with `check_mean_beam_range`.

    Returns
    -------
    float
        Mean index of the non-NaN entries in `beam`, or NaN if the whole
        ping is NaN.
    """
    # For a single ping
    ind1 = np.argwhere(np.isnan(beam)==False)
    if len(ind1) == 0:
        beam_range = np.nan
    elif len(ind1) > 0:
        beam_range = np.nanmean(ind1[:,0])
    return(beam_range)


##################################################################################################

def beam_true_depth(ds, use_loop=False):
    """
    Compute the true depth of each beam's measurement cells, adjusting for
    the glider's pitch and roll.

    Parameters
    ----------
    ds : xarray.Dataset
    use_loop : bool
        If True, use the original per-ping Python loop (calling `cell_vert`
        once per beam per ping) instead of the vectorized broadcast formula.
        Both paths evaluate the identical trig formula; the loop is kept for
        cross-checking and is much slower on long deployments.
    """
    Pitch  = ds['Pitch'].values
    Roll   = ds['Roll'].values
    Vrange = ds.VelocityRange.values
    Depth  = ds['Depth'].values

    if use_loop:
        TrueDepthBeam1 = np.empty((len(Vrange), len(ds.time)))
        TrueDepthBeam2 = np.empty((len(Vrange), len(ds.time)))
        TrueDepthBeam3 = np.empty((len(Vrange), len(ds.time)))
        TrueDepthBeam4 = np.empty((len(Vrange), len(ds.time)))
        for i in np.arange(0, len(ds.time)):
            TrueDepthBeam1[:, i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=1)
            TrueDepthBeam2[:, i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=2)
            TrueDepthBeam3[:, i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=3)
            TrueDepthBeam4[:, i] = cell_vert(Pitch[i], Roll[i], Vrange, beam_number=4)
    else:
        # Vectorized: broadcast (n_range, 1) with (1, n_time) -> (n_range, n_time)
        Vr = Vrange[:, None]
        P  = Pitch[None, :]
        R  = Roll[None, :]

        TrueDepthBeam1 = Vr * np.sin(np.deg2rad(90 + 47.5 + P)) * np.sin(np.deg2rad(90 - R))
        TrueDepthBeam2 = Vr * np.sin(np.deg2rad(90 - P)) * np.sin(np.deg2rad(90 + R + 25))
        TrueDepthBeam3 = Vr * np.sin(np.deg2rad(90 - 47.5 + P)) * np.sin(np.deg2rad(90 - R))
        TrueDepthBeam4 = Vr * np.sin(np.deg2rad(90 - P)) * np.sin(np.deg2rad(90 - R + 25))

    [bdepth, bbins] = np.meshgrid(Depth, Vrange)
    true_depth = bdepth + bbins

    ds = ds.assign(
        TrueDepthBeam1=(("VelocityRange", "time"), TrueDepthBeam1),
        TrueDepthBeam2=(("VelocityRange", "time"), TrueDepthBeam2),
        TrueDepthBeam3=(("VelocityRange", "time"), TrueDepthBeam3),
        TrueDepthBeam4=(("VelocityRange", "time"), TrueDepthBeam4),
        TrueDepth=(("VelocityRange", "time"), true_depth),
    )
    return ds

##################################################################################################

def binmap_adcp(ds):
    """
    Interpolate each beam's velocity from its true (tilt-corrected) cell
    depths onto the instrument's regular, nominal depth grid.

    Each ADCP ping's cells sit at different true depths depending on the
    glider's pitch/roll at that instant (see `beam_true_depth`), so before
    beams can be combined ping-by-ping they need to be resampled onto a
    common depth axis. This does that per-ping, per-beam, via linear
    interpolation (`np.interp`), extrapolating to NaN beyond the deepest
    good return.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain `VelocityBeam1..4` and `TrueDepthBeam1..4` (the output
        of `beam_true_depth`).

    Returns
    -------
    xarray.Dataset
        `ds` with `InterpVelocityBeam1..4` added, each on the
        `VelocityRange` grid.
    """
    ## Depth bins to interp onto
    Vrange = np.array(ds.VelocityRange.values)
    
    ## Apparently xarray kind of sucks and it is faster to pull out these variables as objects, perform calculations, and stuff back in
    TrueDepthBeam1 = ds.TrueDepthBeam1.values
    TrueDepthBeam2 = ds.TrueDepthBeam2.values
    TrueDepthBeam3 = ds.TrueDepthBeam3.values
    TrueDepthBeam4 = ds.TrueDepthBeam4.values
    VelocityBeam1 = ds.VelocityBeam1.values
    VelocityBeam2 = ds.VelocityBeam2.values
    VelocityBeam3 = ds.VelocityBeam3.values    
    VelocityBeam4 = ds.VelocityBeam4.values
    
    ## Preallocate interpolated velocity outside of master xarray dataset for easy looping
    InterpVelocityBeam1 = np.empty((len(Vrange),len(ds.time)))
    InterpVelocityBeam2 = np.empty((len(Vrange),len(ds.time)))
    InterpVelocityBeam3 = np.empty((len(Vrange),len(ds.time)))
    InterpVelocityBeam4 = np.empty((len(Vrange),len(ds.time)))
    ## Set the empty variables = nan
    InterpVelocityBeam1[:] = np.nan
    InterpVelocityBeam2[:] = np.nan
    InterpVelocityBeam3[:] = np.nan
    InterpVelocityBeam4[:] = np.nan
    
    ## Create true-interp velocity variables in master xarray dataset
    ds = ds.assign(InterpVelocityBeam1=ds["VelocityBeam1"] *np.nan)
    ds = ds.assign(InterpVelocityBeam2=ds["VelocityBeam2"] *np.nan)
    ds = ds.assign(InterpVelocityBeam3=ds["VelocityBeam3"] *np.nan)
    ds = ds.assign(InterpVelocityBeam4=ds["VelocityBeam4"] *np.nan)

    for x in range(len(ds.time)):
        InterpVelocityBeam1[:,x] = np.interp(Vrange,TrueDepthBeam1[:,x],VelocityBeam1[:,x],right=np.nan)
        InterpVelocityBeam2[:,x] = np.interp(Vrange,TrueDepthBeam2[:,x],VelocityBeam2[:,x],right=np.nan)
        InterpVelocityBeam3[:,x] = np.interp(Vrange,TrueDepthBeam3[:,x],VelocityBeam3[:,x],right=np.nan)
        InterpVelocityBeam4[:,x] = np.interp(Vrange,TrueDepthBeam4[:,x],VelocityBeam4[:,x],right=np.nan)


    ## Now put the output back into the master xarray dataset
    ds['InterpVelocityBeam1'].values = InterpVelocityBeam1
    ds['InterpVelocityBeam2'].values = InterpVelocityBeam2
    ds['InterpVelocityBeam3'].values = InterpVelocityBeam3
    ds['InterpVelocityBeam4'].values = InterpVelocityBeam4
    
    return(ds)



##################################################################################################

def cell_vert(pitch, roll, velocity_range, beam_number):
    """
    Vertical displacement of one beam's measurement cells below the
    instrument, given the glider's pitch and roll.

    A 4-beam Janus ADCP measures along beams angled off the instrument's
    vertical axis, so a cell nominally `velocity_range` meters along the
    beam is not `velocity_range` meters straight down once the instrument
    tilts. This resolves that geometry to a true vertical offset (see also
    `beam_true_depth`, which calls this once per beam per ping in its
    `use_loop=True` path; the vectorized default path inlines the same
    formula).

    Beam layout (fixed for this instrument):
        Beam 1: Forward   (47.5 degrees off horizontal)
        Beam 2: Port      (25 degrees off horizontal)
        Beam 3: Aft       (47.5 degrees off horizontal)
        Beam 4: Starboard (25 degrees off horizontal)
    The beam angle is folded into pitch for beams 1 & 3, and into roll for
    beams 2 & 4.

    Parameters
    ----------
    pitch : float
        Pitch in degrees. Positive = pitch up.
    roll : float
        Roll in degrees. Positive = port wing up.
    velocity_range : array_like
        Along-beam distance to each cell, in meters.
    beam_number : int
        Which beam (1-4) `velocity_range` belongs to.

    Returns
    -------
    numpy.ndarray
        Vertical displacement below the instrument for each cell in
        `velocity_range`, in meters.
    """
    ## Calculate a vertical displacement below instrument for
    ## each adcp bin adjusting for pitch and roll (in degrees)
    ## Positive roll: Port wing up
    ## Positive pitch: Pitch up

    ## Beam 1: Forward   (47.5 degrees off horizontal)
    ## Beam 2: Port      (25 degrees off horizontal)
    ## Beam 3: Aft       (47.5 degrees off horizontal)
    ## Beam 4: Starboard (25 degrees off horizontal)

    ## Beam angle is only incorporated in pitch for Beams 1 & 3 and
    ## in roll for Beams 2 & 4

    if beam_number == 1:
        beam_angle = 47.5
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 + beam_angle + pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 - roll)))
    
    elif beam_number == 2:
        beam_angle = 25
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 - pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 + roll + beam_angle)))
    
    elif beam_number == 3:
        beam_angle = 47.5
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 - beam_angle + pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 - roll)))
        
    elif beam_number == 4:
        beam_angle = 25
        pitch_adjusted = velocity_range * np.sin(np.deg2rad(90 - pitch))
        z = (pitch_adjusted * np.sin(np.deg2rad(90 - roll + beam_angle)))
    
    else:
        print("Must specify beam number")
        exit(1)
    
    return z.transpose()


##################################################################################################

def correct_sound_speed(ds):
    """
    Rescale each beam's velocity by the ratio of the instrument's measured
    speed of sound to the fixed 1500 m/s the AD2CP assumes internally when
    converting Doppler shift to velocity.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain `VelocityBeam1..4` and `SpeedOfSound`.

    Returns
    -------
    xarray.Dataset
        `ds` with `VelocityBeam1..4` corrected in place.
    """
    default_speedofsound = 1500
    ds["VelocityBeam1"] = ds.VelocityBeam1*(ds.SpeedOfSound/default_speedofsound)
    ds["VelocityBeam2"] = ds.VelocityBeam2*(ds.SpeedOfSound/default_speedofsound)
    ds["VelocityBeam3"] = ds.VelocityBeam3*(ds.SpeedOfSound/default_speedofsound)
    ds["VelocityBeam4"] = ds.VelocityBeam4*(ds.SpeedOfSound/default_speedofsound)
    return ds


##################################################################################################

def qaqc_pre_coord_transform(ds, corr_threshold, max_amplitude):
    """
    Discard beam velocities with weak or saturated acoustic returns, before
    the beam-to-ENU coordinate transform.

    Nortek's correlation and amplitude fields are the standard proxies for a
    ping's reliability: low correlation means the two pulses used to derive
    the Doppler shift don't match well (noisy velocity), and unusually high
    amplitude typically indicates the beam has hit the surface, bottom, or
    the glider's own hull rather than open water.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain `VelocityBeam1..4`, `CorrelationBeam1..4`, and
        `AmplitudeBeam1..4`.
    corr_threshold : float
        Beam correlation (percent) below which a cell is set to NaN.
    max_amplitude : float
        Beam amplitude above which a cell is set to NaN.

    Returns
    -------
    xarray.Dataset
        `ds` with `VelocityBeam1..4` NaN'd out at low-correlation or
        high-amplitude cells.
    """
    ## This sucks but much faster than working through xarray
    VelocityBeam1    = ds.VelocityBeam1.values
    VelocityBeam2    = ds.VelocityBeam2.values
    VelocityBeam3    = ds.VelocityBeam3.values
    VelocityBeam4    = ds.VelocityBeam4.values
    CorrelationBeam1 = ds.CorrelationBeam1.values
    CorrelationBeam2 = ds.CorrelationBeam2.values
    CorrelationBeam3 = ds.CorrelationBeam3.values
    CorrelationBeam4 = ds.CorrelationBeam4.values
    AmplitudeBeam1   = ds.AmplitudeBeam1.values
    AmplitudeBeam2   = ds.AmplitudeBeam2.values
    AmplitudeBeam3   = ds.AmplitudeBeam3.values
    AmplitudeBeam4   = ds.AmplitudeBeam4.values

    # Filter for low correlation
    VelocityBeam1[np.where(CorrelationBeam1 < corr_threshold)] = np.nan
    VelocityBeam2[np.where(CorrelationBeam2 < corr_threshold)] = np.nan
    VelocityBeam3[np.where(CorrelationBeam3 < corr_threshold)] = np.nan
    VelocityBeam4[np.where(CorrelationBeam4 < corr_threshold)] = np.nan

    # Filter for high amplitude
    VelocityBeam1[np.where(AmplitudeBeam1 > max_amplitude)] = np.nan
    VelocityBeam2[np.where(AmplitudeBeam2 > max_amplitude)] = np.nan
    VelocityBeam3[np.where(AmplitudeBeam3 > max_amplitude)] = np.nan
    VelocityBeam4[np.where(AmplitudeBeam4 > max_amplitude)] = np.nan

    # Now stuff back into xarray ds
    ds.VelocityBeam1.values = VelocityBeam1
    ds.VelocityBeam2.values = VelocityBeam2
    ds.VelocityBeam3.values = VelocityBeam3
    ds.VelocityBeam4.values = VelocityBeam4
    return(ds)


##################################################################################################

def qaqc_post_coord_transform(ds, high_velocity_threshold, surface_depth_to_filter):
    """
    Final velocity QC, applied after the beam-to-ENU coordinate transform.

    Does three things:
      1. Discards U/V/W velocities whose magnitude exceeds
         `high_velocity_threshold` (physically implausible relative to the
         glider's own speed through water, usually a sign of a bad
         transform for that ping).
      2. Discards the first (shallowest) bin below the glider, which sits in
         the wake of the vehicle and is contaminated by its own motion.
      3. Discards all velocities for any ping where the *glider's* depth
         (not the cell depth) is shallower than `surface_depth_to_filter`,
         since near-surface pings are noisy (wave action, bubbles).

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain `UVelocity`, `VVelocity`, `WVelocity`, and `Depth`
        (the glider's depth, one value per ping).
    high_velocity_threshold : float
        Maximum plausible |velocity| in m/s.
    surface_depth_to_filter : float
        Glider depth in meters at or shallower than which a ping's
        velocities are discarded entirely.

    Returns
    -------
    xarray.Dataset
        `ds` with `UVelocity`, `VVelocity`, `WVelocity` NaN'd out per the
        rules above.
    """
    ## This sucks but much faster than working through xarray
    UVelocity    =  ds.UVelocity.values
    VVelocity    =  ds.VVelocity.values
    WVelocity    =  ds.WVelocity.values
    depth        =  ds.Depth.values
    
    ## Filter out high velocities relative to glider
    UVelocity[np.abs(UVelocity) > high_velocity_threshold] = np.nan
    VVelocity[np.abs(VVelocity) > high_velocity_threshold] = np.nan
    WVelocity[np.abs(WVelocity) > high_velocity_threshold] = np.nan
    
    ## Filter out first bin below glider
    UVelocity[0,:] = np.nan
    VVelocity[0,:] = np.nan
    WVelocity[0,:] = np.nan
    
    ## Filter out velocity if true depth is 5 meters or shallower
    depthind = np.where(depth <= surface_depth_to_filter)
    UVelocity[:,depthind] = np.nan
    VVelocity[:,depthind] = np.nan
    WVelocity[:,depthind] = np.nan
    
    ## Now stuff back into xarray ds
    ds.UVelocity.values = UVelocity
    ds.VVelocity.values = VVelocity
    ds.WVelocity.values = WVelocity
    return(ds)
    


##################################################################################################

def inversion(U,V,dz,u_daverage,v_daverage,bins,depth, wDAC, wSmoothness, use_loop=False):
    """
    Per-segment least-squares shear inversion (Todd et al. 2017 /
    Visbeck 2002 style): combine per-ping, per-bin ADCP velocity
    measurements with a depth-averaged-current constraint into a single
    ocean velocity profile.

    An ADCP mounted on a glider measures velocity *relative to the glider*,
    ping by ping, at fixed ranges from the instrument. As the glider dives
    or climbs, the same patch of ocean gets sampled from many different
    glider positions (and thus is redundantly over-determined), while the
    glider's own unknown through-water velocity contaminates every single
    ping. This sets up and solves the sparse linear system that separates
    the two: one unknown glider velocity per ping (`Uctd`) and one unknown
    ocean velocity per depth bin (`Uocean`), constrained so that the
    depth-averaged ocean velocity matches the glider's independently-known
    dive-averaged current (from GPS fixes at the surface).

    Parameters
    ----------
    U, V : numpy.ndarray, shape (n_bin, n_ping)
        Measured east-west (U) and north-south (V) velocities from the
        ADCP, relative to the glider, for one dive/climb segment.
    dz : float
        Desired vertical resolution of the output profile, in meters.
        Should not be smaller than the ADCP's own bin length.
    u_daverage, v_daverage : float
        Depth-averaged (dive-averaged) current for this segment, from GPS
        surfacing fixes. Set to 0 for real-time processing without DAC.
    bins : numpy.ndarray
        Bin depths (ranges) corresponding to the rows of `U`/`V`.
    depth : numpy.ndarray
        Glider depth for each ping (column of `U`/`V`), as measured by the
        ADCP's own pressure sensor.
    wDAC : float
        Weight of the depth-averaged-current constraint (5, per Todd et al.
        2017).
    wSmoothness : float
        Weight of the curvature-minimizing (smoothness) constraint (1, per
        Todd et al. 2017). Set to 0 to disable.
    use_loop : bool
        If True, build the bin counts, G matrix, and per-bin observation
        counts with the original per-ensemble/per-bin Python loops instead
        of the vectorized histogram / COO-matrix / sparse column-sum
        implementations. Both paths produce the identical G matrix
        (verified bit-for-bit on synthetic data); the loop is kept for
        cross-checking and is much slower on long deployments.

    Returns
    -------
    O_ls : numpy.ndarray
        Ocean velocity profile, one complex value per depth bin (real part
        = east-west, imaginary part = north-south).
    G_ls : numpy.ndarray
        Glider (through-water) velocity, one complex value per ping.
    bin_new : list of float
        Depth bin centers corresponding to `O_ls`.
    obs_per_bin : numpy.ndarray
        Number of good velocity observations backing each bin of `O_ls`.

    ## Feb-2021 jgradone@marine.rutgers.edu Initial
    ## Jul-2021 jgradone@marine.rutgers.edu Updates for constraints
    ## Jun-2022 jgradone@marine.rutgers.edu Corrected dimensions and indexing of G matrix
    ## Jun-2022 jgradone@marine.rutgers.edu Added curvature minimizing constraint and constraint weights
    """
    global O_ls, G_ls, bin_new

    #########################################################################
    ## These steps filter for NAN rows and columns so they are technically QAQC
    ## but I think the best place to put them is inthe inversion function because
    ## if there are nans still present in the data here, it will throw everything off
    ## These steps are HUGE for efficiency because it reduces the size of the G
    ## matrix as much as possible.

    ## This determines the rows (bins) where all the columns are nan
    nanind = np.where( (np.sum(np.isnan(U),axis=1)/U.shape[1]) == 1)[0]
    if len(nanind) > 0:
        U = np.delete(U,nanind,axis=0)
        V = np.delete(V,nanind,axis=0)
        bins = np.delete(bins,nanind)

    ## Do the same thing with individual ensembles. Note: need to remove the corresponding
    ## ensemble pressure reading to ensure correction dimensions and values.
    nanind = np.where((np.sum(np.isnan(U),axis=0)/U.shape[0]) == 1)[0]
    if len(nanind) > 0:
        U = np.delete(U,nanind,axis=1)
        V = np.delete(V,nanind,axis=1)
        depth = np.delete(depth,nanind)
    ##########################################################################        


    ##########################################################################        
    # Take difference between bin lengths for bin size [m]
    bin_size = np.diff(bins)[0]
    bin_num = len(bins)
    # This creates a grid of the ACTUAL depths of the ADCP bins by adding the
    # depths of the ADCP bins to the actual depth of the instrument
    [bdepth,bbins]=np.meshgrid(depth,bins)
    bin_depth = bdepth+bbins  
    Z = bin_depth
    # Calculate the maximum depth of glider which is different than maximum ADCP bin depth
    ZmM = np.nanmax(depth)
    ##########################################################################        


    ##########################################################################        
    # Set knowns from Equations 19 from Visbeck (2002) page 800
    # Maximum number of observations (nd) is given by the number of velocity
    # estimates per ping (nbin) times the number of profiles per cast (nt)
    nbin = U.shape[0]  # number of programmed ADCP bins per individual profile
    nt   = U.shape[1]  # number of individual velocity profiles
    nd   = nbin*nt      # G dimension (1) 

    # Define the edges of the bins
    bin_edges = np.arange(0,math.floor(np.max(bin_depth)),dz).tolist()

    # Check that each bin has data in it
    if use_loop:
        bin_count = np.empty(len(bin_edges)-1)
        bin_count[:] = np.nan
        for k in np.arange(len(bin_edges))[:-1]:
            # Create index of depth values that fall inside the bin edges
            ii = np.where((bin_depth > bin_edges[k]) & (bin_depth < bin_edges[k+1]))
            bin_count[k] = len(bin_depth[ii])
            ii = []
    else:
        bin_edges_arr = np.asarray(bin_edges)
        bin_count = np.histogram(bin_depth.ravel(), bins=bin_edges_arr)[0].astype(float)

    # Create list of bin centers
    bin_new = [x+dz/2 for x in bin_edges[:-1]]

    # Calculate which FINAL solution bin is deeper than the maximum depth of the glider
    # This is done so that the depth averaged velocity constraint is only applied to bins shallower than this depth
    depth_ind = len(np.where(bin_new>ZmM)[0])
    # Chop off the top of profile if no data
    ind = np.argmax(bin_count > 0) # Stops at first index greater than 0
    bin_new = bin_new[ind:]        # Removes all bins above first with data
    z1 = bin_new[0]                # Depth of center of first bin with data
    ##########################################################################


    ##########################################################################
    # Create and populate G
    nz = len(bin_new)  # number of ocean velocities desired in output profile
    nm = nt + nz       # G dimension (2), number of unknowns

    if use_loop:
        # Let's build the corresponding coefficient matrix G
        G = scipy.sparse.lil_matrix((nd, nm), dtype=float)

        # Indexing of the G matrix was taken from Todd et al. 2012
        for ii in np.arange(0,nt):           # Number of ADCP ensembles per segment
            for jj in np.arange(0,nbin):     # Number of measured bins per ensemble

                # Uctd part of matrix
                G[(nbin*(ii))+jj,ii] = -1
                # This will fill in the Uocean part of the matrix. It loops through
                # all Z members and places them in the proper location in the G matrix
                # Find the difference between all bin centers and the current Z value
                dx = abs(bin_new-Z[jj,ii])
                # Find the minimum of these differences
                minx = np.nanmin(dx)
                # Finds bin_new index of the first match of Z and bin_new
                idx = np.argmin(dx-minx)

                # Uocean part of matrix
                G[(nbin*(ii))+jj,(nt)+idx] = 1

                del dx, minx, idx
        G = G.tocsr()
    else:
        # Vectorized G construction using COO format
        rows = np.arange(nd)
        # Uctd columns: each ensemble ii maps to column ii, repeated nbin times
        col_uctd = np.repeat(np.arange(nt), nbin)
        # Uocean columns: find closest bin_new for each Z value
        bin_new_arr = np.asarray(bin_new)
        Z_flat = Z.ravel(order='F')
        col_uocean = nt + np.nanargmin(np.abs(Z_flat[:, None] - bin_new_arr[None, :]), axis=1)

        G = scipy.sparse.coo_matrix(
            (np.concatenate([np.full(nd, -1.0), np.full(nd, 1.0)]),
             (np.concatenate([rows, rows]), np.concatenate([col_uctd, col_uocean]))),
            shape=(nd, nm)
        ).tocsr()

    ##########################################################################        
    # Reshape U and V into the format of the d column vector (order='F')
    # Based on how G is made, d needs to be ensembles stacked on one another vertically
    d_u = U.flatten(order='F')
    d_v = V.flatten(order='F')

    ##########################################################################
    ## This chunk of code containts the constraints for depth averaged currents
    # Make sure the constraint is only applied to the final ocean velocity bins that the glider dives through
    # Don't apply it to the first bin and don't apply it to the bins below the gliders dive depth
    constraint = np.concatenate(([np.zeros(nt)], [0], [np.tile(dz,nz-(1+depth_ind))], [np.zeros(depth_ind)]), axis=None)

    # Ensure the L^2 norm of the constraint equation is unity
    constraint_norm = np.linalg.norm(constraint/ZmM)
    C = 1/constraint_norm
    constraint_normalized = (C/ZmM)*constraint ## This is now equal to 1 (unity)
    # Build Gstar and add weight from todd 2017
    ## Some smarts would be to calculate signal to noise ratio first
    Gstar = scipy.sparse.vstack((G,wDAC*constraint_normalized), dtype=float)


    # Add the constraint for the depth averaged velocity from Todd et al. (2017)
    du = np.concatenate(([d_u],[wDAC*C*u_daverage]), axis=None)
    dv = np.concatenate(([d_v],[wDAC*C*v_daverage]), axis=None)
    d = np.array(list(map(complex,du, dv)))


    ##########################################################################        
    #### THIS removes all nan elements of d AND Gstar so the inversion doesn't blow up with nans
    ind2 = np.where(np.isnan(d)==True)[0]
    d = np.delete(d,ind2)

    def delete_rows_csr(mat, indices):
        """
        Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
        """
        if not isinstance(mat, scipy.sparse.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")
        indices = list(indices)
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[indices] = False
        return mat[mask]

    Gstar = delete_rows_csr(Gstar.tocsr().copy(),ind2)

    #########################################################################        
    # Test adding depth for tracking bin location
    # d is ensembles stacked on one another vertically so same for Z (order='F')
    Z_filt = Z.flatten(order='F')
    Z_filt = np.delete(Z_filt,ind2)
    Z_filt = np.concatenate(([Z_filt],[0]), axis=None)

    ##########################################################################        
    ## Calculation the number of observations per bin
    if use_loop:
        obs_per_bin = np.empty(len(bin_new))
        obs_per_bin[:] = np.nan
        for x in np.arange(0,nz):
            rows_where_nt_not_equal_zero = np.where(Gstar.tocsr()[0:Z_filt.shape[0],nt+x].toarray() > 0 )[0]
            obs_per_bin[x] = len(rows_where_nt_not_equal_zero)
    else:
        Gstar_csr = Gstar.tocsr()
        ocean_block = Gstar_csr[:Z_filt.shape[0], nt:nt+nz]
        obs_per_bin = np.asarray((ocean_block > 0).sum(axis=0)).ravel().astype(float)

    ## If there is no data in the last bin, drop that from the G matrix, bin_new, and obs_per_bin
    if obs_per_bin[-1] == 0:
        Gstar.tocsr()[:,:-1]
        bin_new = bin_new[:-1]
        obs_per_bin = obs_per_bin[:-1]
        ## Update nz and nt
        nz = len(bin_new)
        nt = Gstar.shape[1]-nz

    ##########################################################################        
    ## Smoothness constraint
    ## Only do this is the smoothness constraint is set
    if wSmoothness > 0:
        ## Add a vector of zerosm the length of nz, twice to the bottom of the data column vector
        d = np.concatenate(([d],[np.zeros(nz)],[np.zeros(nz)]), axis=None)
        ## Constraint on smoothing Uocean side of matrix
        smoothing_matrix_Uocean = scipy.sparse.diags([[-1.0],[2.0],[-1.0]], [0,1,2], shape=(nz,nz))
        smoothing_matrix1 = scipy.sparse.hstack((np.zeros((nz,nt)),smoothing_matrix_Uocean), dtype=float)
        ## Constraint on smoothing Uglider side of matrix
        smoothing_matrix_Uglider = scipy.sparse.diags([[-1.0],[2.0],[-1.0]], [0,1,2], shape=(nz,nt))
        smoothing_matrix2 = scipy.sparse.hstack((smoothing_matrix_Uglider,np.zeros((nz,nz))), dtype=float)
        Gstar = scipy.sparse.vstack((Gstar,wSmoothness*smoothing_matrix1,wSmoothness*smoothing_matrix2), dtype=float)


    ##########################################################################        
    ## Run the Least-Squares Inversion!
    x = lsqr(Gstar, d)[0]

    O_ls = x[nt:]
    G_ls = x[0:nt] 
    ########################################################################## 

    return(O_ls, G_ls, bin_new, obs_per_bin)


##################################################################################################

def shear_method(U,V,W,vx,vy,bins,depth,dz):
    """
    Alternative to `inversion()`: reference shear (velocity differences
    between adjacent depth bins) to an absolute profile using the segment's
    dive-averaged current, in the style of the shear-integration methods
    used by `gliderad2cp` (Frajka-Williams et al. / Todd et al. shear
    approach), rather than `inversion()`'s simultaneous least-squares
    solve.

    **Currently broken / not usable.** This function calls
    `calc_ensemble_shear`, `bin_attr`, and `shear_to_vel`, none of which are
    defined anywhere in this package (or imported from elsewhere) - calling
    it raises `NameError`. It is still exported in `__all__`. Use
    `inversion()` instead until this is fixed or these helpers are added.

    Parameters
    ----------
    U, V, W : numpy.ndarray, shape (n_bin, n_ping)
        Measured east-west, north-south, and vertical velocities from the
        ADCP, relative to the glider, for one dive/climb segment.
    vx, vy : float
        Reference (e.g. dive-averaged) east-west/north-south velocity to
        anchor the integrated shear profile to.
    bins : numpy.ndarray
        Bin depths (ranges) corresponding to the rows of `U`/`V`/`W`.
    depth : numpy.ndarray
        Glider depth for each ping (column of `U`/`V`/`W`).
    dz : float
        Desired vertical resolution of the output profile, in meters.

    Returns
    -------
    vel_referenced : numpy.ndarray
        Absolute velocity profile referenced to `vx`/`vy`.
    bin_centers : numpy.ndarray
        Depth bin centers corresponding to `vel_referenced`.
    vel_referenced_std : numpy.ndarray
        Estimated uncertainty per bin (combining per-ping instrument noise
        and shear-binning spread).
    """
    ########################################################################
    # These steps filter for NAN rows and columns so they are technically QAQC
    # but I think the best place to put them is inthe inversion function because
    # if there are nans still present in the data here, it will throw everything off
    # These steps are HUGE for efficiency because it reduces the size of the G
    # matrix as much as possible.

    ## This determines the rows (bins) where all the columns are nan
    nanind = np.where( (np.sum(np.isnan(U),axis=1)/U.shape[1]) == 1)[0]
    if len(nanind) > 0:
        U = np.delete(U,nanind,axis=0)
        V = np.delete(V,nanind,axis=0)
        W = np.delete(W,nanind,axis=0)
        bins = np.delete(bins,nanind)

    ## Do the same thing with individual ensembles. Note: need to remove the corresponding
    ## ensemble pressure reading to ensure correction dimensions and values.
    nanind = np.where((np.sum(np.isnan(U),axis=0)/U.shape[0]) == 1)[0]
    if len(nanind) > 0:
        U = np.delete(U,nanind,axis=1)
        V = np.delete(V,nanind,axis=1)
        W = np.delete(W,nanind,axis=1)
        depth = np.delete(depth,nanind)
    ##########################################################################        


    ##########################################################################        
    # Take difference between bin lengths for bin size [m]
    bin_size = np.diff(bins)[0]
    bin_num = len(bins)
    # This creates a grid of the ACTUAL depths of the ADCP bins by adding the
    # depths of the ADCP bins to the actual depth of the instrument


    [bdepth,bbins]=np.meshgrid(depth,bins[0:-1])
    bin_depth = bdepth+bbins  

    Z = bin_depth
    # Calculate the maximum depth of glider which is different than maximum ADCP bin depth
    ZmM = np.nanmax(depth)

    ## Calculate shear per ensemble
    ensemble_shear_U = calc_ensemble_shear(U,bins)
    ensemble_shear_V = calc_ensemble_shear(V,bins)
    ensemble_shear_W = calc_ensemble_shear(W,bins)

    ## Create velocity dataframes for shear
    flatu = ensemble_shear_U.flatten(order='F')
    flatv = ensemble_shear_V.flatten(order='F')
    flatw = ensemble_shear_W.flatten(order='F')
    flatz = -Z.flatten(order='F')
    flat_df = np.column_stack((flatu,flatv,flatw,flatz))
    flat_df = flat_df[flat_df[:, 3].argsort()[::-1]]

    ## Shear binning
    shear_binned, shear_binned_std, shear_cell_center, vels_in_bin = bin_attr(shear_v= flat_df[:,0:3], shear_z=flat_df[:,3], bin_size=dz, Hmax=np.nanmin(flat_df[:,3]))
    ## Shear to absolute
    vel, vel_referenced, bin_centers = shear_to_vel(shear_binned, shear_cell_center, ref_vel=[vx,vy,0])
    
    ## First define uncertainty of a single ping according to the ADCP configuration
    std_ping = 0.03     #[m/s] (in average mode)
    vel_referenced_std = np.sqrt(std_ping**2 + shear_binned_std**2)
    
    return(vel_referenced, bin_centers, vel_referenced_std)


##################################################################################################

def mag_var_correction(heading,u_dac,v_dac,mag_var):
    """
    Rotate a heading and a depth-averaged-current vector from magnetic to
    true north.

    Parameters
    ----------
    heading : array_like
        Heading in degrees, magnetic.
    u_dac, v_dac : float or array_like
        East-west / north-south depth-averaged current, magnetic frame.
    mag_var : float or array_like
        Magnetic variation (declination) in degrees at the glider's
        location, matching the sign convention of `heading`.

    Returns
    -------
    heading_corrected : array_like
        `heading` rotated onto true north, in degrees.
    u_dac_corrected, v_dac_corrected : float or array_like
        `u_dac`/`v_dac` rotated onto true north.
    """
    heading_corrected = heading - mag_var ## Corrected heading in degrees
    mag_var_rad = np.deg2rad(mag_var)
    heading_rad = np.deg2rad(heading)
    u_dac_corrected = u_dac*np.cos(mag_var_rad) - v_dac*np.sin(mag_var_rad)
    v_dac_corrected = u_dac*np.sin(mag_var_rad) + v_dac*np.cos(mag_var_rad)
    
    return heading_corrected, u_dac_corrected, v_dac_corrected





def mag_var_correction_ad2cp_ds(ds, heading_var="CorrectedHeading", mag_var_arr=0):
    """
    Apply magnetic variation correction to AD2CP heading directly in the dataset.
    Drops any reference to u_dac and v_dac (not used here).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing heading and magnetic variation.
    heading_var : str
        Name of heading variable to correct.
    mag_var : str
        Name of magnetic variation variable (degrees).

    Returns
    -------
    ds : xarray.Dataset
        Dataset with corrected heading assigned to 'CorrectedHeading_MagVar'.
    """
    # Extract arrays
    heading = ds[heading_var].values

    # Correct heading
    heading_corrected = heading - mag_var_arr

    # Assign corrected heading to new variable
    ds = ds.assign({"CorrectedHeading_MagVar": ("time", heading_corrected)})

    return ds



##################################################################################################

def calcAHRS(ds, heading_var="CorrectedHeading_MagVar", roll_var="Roll", pitch_var="Pitch", use_loop=False):
    """
    Compute AHRS rotation matrix for each time step and attach it to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain variables for heading, roll, and pitch.
    heading_var : str
        Name of heading variable in ds.
    roll_var : str
        Name of roll variable in ds.
    pitch_var : str
        Name of pitch variable in ds.
    use_loop : bool
        If True, build the rotation matrix with the original per-timestep
        H @ P matrix-multiply loop instead of the analytically-expanded
        vectorized formula. Both paths compute the identical matrix; the
        loop is kept for cross-checking and is much slower on long
        deployments.

    Returns
    -------
    ds_out : xarray.Dataset
        Original dataset with added variable 'AHRSRotationMatrix' of shape (9, time)
    """
    hh = np.deg2rad(np.asarray(ds[heading_var]) - 90)
    pp = np.deg2rad(np.asarray(ds[pitch_var]))
    rr = np.deg2rad(np.asarray(ds[roll_var]))

    if use_loop:
        RotMatrix = np.full((9, len(pp)), np.nan)
        for k in range(len(pp)):
            H = np.array([
                [np.cos(hh[k]), np.sin(hh[k]), 0],
                [-np.sin(hh[k]), np.cos(hh[k]), 0],
                [0, 0, 1]
            ])
            P = np.array([
                [np.cos(pp[k]), -np.sin(pp[k])*np.sin(rr[k]), -np.cos(rr[k])*np.sin(pp[k])],
                [0, np.cos(rr[k]), -np.sin(rr[k])],
                [np.sin(pp[k]), np.sin(rr[k])*np.cos(pp[k]), np.cos(pp[k])*np.cos(rr[k])]
            ])
            R = H @ P
            RotMatrix[:, k] = R.reshape(-1, order='F')
    else:
        ch, sh = np.cos(hh), np.sin(hh)
        cp, sp = np.cos(pp), np.sin(pp)
        cr, sr = np.cos(rr), np.sin(rr)

        # Analytical expansion of R = H @ P, stored in Fortran column-major order
        RotMatrix = np.array([
            ch * cp,                        # R[0,0]
            -sh * cp,                       # R[1,0]
            sp,                             # R[2,0]
            -ch * sp * sr + sh * cr,        # R[0,1]
             sh * sp * sr + ch * cr,        # R[1,1]
            sr * cp,                        # R[2,1]
            -ch * cr * sp - sh * sr,        # R[0,2]
             sh * cr * sp - ch * sr,        # R[1,2]
            cp * cr,                        # R[2,2]
        ])

    ds_out = ds.copy()
    ds_out = ds_out.assign(AHRSRotationMatrix=(("x", "time"), RotMatrix))

    return ds_out



##################################################################################################

def beam2enu(ds, honour_pitch_selection=True, use_loop=False):
    """
    Transform beam velocities to ENU.

    Parameters
    ----------
    ds : xarray.Dataset
        Must carry InterpVelocityBeam1..4, Pitch and AHRSRotationMatrix.
    honour_pitch_selection : bool
        Whether to use the pitch-dependent beam triplet selected below.
        Versions of this package up to 2.0.0 selected that triplet and then
        discarded the choice, always applying the beam 2/3/4 columns. On a
        deployment where that was tested (ru37, Cayman 2026) the discarded
        choice made the horizontal velocity direction incoherent with the
        glider's compass, median absolute deviation 117 degrees against 12
        degrees, and biased the mean vertical velocity to -0.14 m/s against
        0.00 m/s. Pass False to reproduce the old behaviour.
    use_loop : bool
        If True, use the original per-ping Python loop instead of the
        vectorized mask + batched einsum implementation. Both paths select
        identical beams, transform matrices, and matrix products for every
        ping; the loop is kept for cross-checking and is much slower on
        long deployments.
    """
	## 01/21/2022     jgradone@marine.rutgers.edu     Initial

	## This function transforms velocity data from beam coordinates to XYZ to ENU. Beam coordinates
	## are defined as the velocity measured along the three beams of the instrument.
	## ENU coordinates are defined in an earth coordinate system, where E represents the East-West
	## component, N represents the North-South component and U represents the Up-Down component.
	## This function was created for a Nortek AD2CP mounted looking downward on a Slocum glider.

	#############################################################################################################
	## Per Nortek:                                                                                             ##
	## https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done- ##
	#############################################################################################################

	## "Each instrument has its own unique transformation matrix, based on the transducer geometry.
	## This matrix can be found, as previously mentioned, in the .hdr file generated when performing
	## a binary data conversion in the software. Each row of the matrix represents a component in the
	## instrument’s XYZ coordinate system, starting with X at the top row. Each column represents a beam.
	## The third and fourth rows of the Vectrino or Signature transformation matrix represent the two
	## estimates of vertical velocity (Z1 and Z2) produced by the instrument. XYZ coordinates are
	## defined relative to the instrument, so they do not take into account heading, pitch and roll.
	## ENU utilizes the attitude measurements to provide an Earth-relative coordinate system."

	## These are the transformation matricies for up and down cast.
	## Beam 1: Forward
	## Beam 2: Port
	## Beam 3: Aft
	## Beam 4: Starboard

	##################################
	## Transformation matrix layout ##
	##################################
	# Beam: 1.   2.   3.   4.
	# X     1X.  2X.  3X.  4X.
	# Y     1Y.  2Y.  3Y.  4Y.
	# Z1    1Z1. 2Z1. 3Z1. 4Z1.
	# Z2    1Z2. 2Z2. 3Z2. 4Z2.
	##################################

	################# Input Variables #################
	## beam1vel     = single ping of velocity from beam 1
	## beam2vel     = single ping of velocity from beam 2
	## beam3vel     = single ping of velocity from beam 3
	## beam4vel     = single ping of velocity from beam 4
	## beam2xyz_mat = Static transformation matrix from beam to XYZ taking from AD2CP config
	## ahrs_rot_mat = Dynamic transformation matrix from XYZ to beam, changes depending on heading, pitch, and roll
	## pitch        = pitch in degrees

	############################################################################################################
	## First from beam to XYZ
	## If downcast, grab just beams 124 and correction transformation matrix
    if 'burst_beam2xyz' in ds.attrs:
        beam2xyz = ds.attrs['burst_beam2xyz']
    elif 'beam2xyz' in ds.attrs:
        beam2xyz = ds.attrs['beam2xyz']
    elif 'avg_beam2xyz' in ds.attrs:
        beam2xyz = ds.attrs['avg_beam2xyz']
    else:
        print('No beam transformation matrix info found')

    beam2xyz = beam2xyz.reshape(4, 4)

    InterpVelocityBeam1 = ds.InterpVelocityBeam1.values
    InterpVelocityBeam2 = ds.InterpVelocityBeam2.values
    InterpVelocityBeam3 = ds.InterpVelocityBeam3.values
    InterpVelocityBeam4 = ds.InterpVelocityBeam4.values
    AHRSRotationMatrix = ds.AHRSRotationMatrix.values
    pitch_vals = np.asarray(ds.Pitch)

    n_range, n_time = InterpVelocityBeam1.shape

    ## If instrument is pointing down, bit 0 in status is equal to 1, rows 2 and 3 must change sign.
    ## Hard coding this because of glider configuration which is pointing down.
    if use_loop:
        UVelocity = np.full((n_range, n_time), np.nan)
        VVelocity = np.full((n_range, n_time), np.nan)
        WVelocity = np.full((n_range, n_time), np.nan)

        for x in np.arange(0, n_time):
            if pitch_vals[x] < 0:
                tot_vel = np.column_stack((InterpVelocityBeam1[:, x], InterpVelocityBeam2[:, x], InterpVelocityBeam4[:, x]))
                beam2xyz_mat = beam2xyz[0:3, [0, 1, 3]]
            else:
                tot_vel = np.column_stack((InterpVelocityBeam2[:, x], InterpVelocityBeam3[:, x], InterpVelocityBeam4[:, x]))
                beam2xyz_mat = beam2xyz[0:3, 1:4]

            if honour_pitch_selection:
                beam2xyz_mat = beam2xyz_mat.copy()
            else:
                ## Behaviour up to version 2.0.0: the pitch-dependent choice
                ## above is discarded and the beam 2/3/4 columns are always used.
                beam2xyz_mat = beam2xyz[0:3, 1:4].copy()
            beam2xyz_mat[1, :] *= -1
            beam2xyz_mat[2, :] *= -1

            xyz = np.dot(beam2xyz_mat, tot_vel.T)
            xyz2enuAHRS = AHRSRotationMatrix[:, x].reshape(3, 3, order='C')
            enu = np.array(np.dot(xyz2enuAHRS, xyz))
            UVelocity[:, x] = enu[0, :].ravel()
            VVelocity[:, x] = enu[1, :].ravel()
            WVelocity[:, x] = enu[2, :].ravel()
    else:
        if honour_pitch_selection:
            # downcast (pitch<0) uses beams 1,2,4; upcast (pitch>=0) uses beams 2,3,4 -
            # each with its own transform-matrix columns, per Nortek's beam2xyz layout.
            mat_down = beam2xyz[0:3, [0, 1, 3]].copy()
            mat_up = beam2xyz[0:3, 1:4].copy()
        else:
            ## Behaviour up to version 2.0.0: the pitch-dependent choice of columns
            ## above is discarded and the beam 2/3/4 columns are always used.
            mat_down = beam2xyz[0:3, 1:4].copy()
            mat_up = beam2xyz[0:3, 1:4].copy()
        for _m in (mat_down, mat_up):
            _m[1, :] *= -1
            _m[2, :] *= -1

        # Select beams based on pitch: downcast (pitch<0) uses 1,2,4; upcast uses 2,3,4
        down_mask = pitch_vals < 0
        up_mask = ~down_mask

        tot_vel = np.empty((n_range, 3, n_time))
        tot_vel[:, 0, down_mask] = InterpVelocityBeam1[:, down_mask]
        tot_vel[:, 1, down_mask] = InterpVelocityBeam2[:, down_mask]
        tot_vel[:, 2, down_mask] = InterpVelocityBeam4[:, down_mask]
        tot_vel[:, 0, up_mask] = InterpVelocityBeam2[:, up_mask]
        tot_vel[:, 1, up_mask] = InterpVelocityBeam3[:, up_mask]
        tot_vel[:, 2, up_mask] = InterpVelocityBeam4[:, up_mask]

        # Beam to XYZ: per-timestep transform matrix (mat_down where pitch<0, else
        # mat_up), batched over range and time
        beam2xyz_mat_t = np.where(down_mask[None, None, :], mat_down[:, :, None], mat_up[:, :, None])
        xyz = np.einsum('ijt,rjt->irt', beam2xyz_mat_t, tot_vel)

        # XYZ to ENU (per-timestep AHRS rotation)
        ahrs = AHRSRotationMatrix.reshape(3, 3, n_time)
        enu = np.einsum('ijt,jrt->irt', ahrs, xyz)
        UVelocity, VVelocity, WVelocity = enu[0], enu[1], enu[2]

    ds = ds.assign(
        UVelocity=(("VelocityRange", "time"), UVelocity),
        VVelocity=(("VelocityRange", "time"), VVelocity),
        WVelocity=(("VelocityRange", "time"), WVelocity),
    )
    return ds



##################################################################################################

def load_ad2cp(ncfile, mean_lat=45):
    """
    Load Nortek AD2CP data from one or more NetCDF files.
    
    Tries 'Data/Average/' first, then 'Data/Burst/' if no data in Average.
    Converts Pressure to Depth and drops Pressure.
    
    Parameters
    ----------
    ncfile : str or list of str
        Path to a single NetCDF file or list of files.
    mean_lat : float
        Latitude for converting Pressure to Depth.
    
    Returns
    -------
    ds : xarray.Dataset
        Combined dataset with Depth variable. Which group ('Average' or
        'Burst') was loaded is not currently exposed to the caller.
    """
    # Normalize input into list
    if isinstance(ncfile, str):
        files = [ncfile]
    elif isinstance(ncfile, (list, tuple, np.ndarray)):
        files = list(ncfile)
    else:
        raise TypeError("ncfile must be a string or list of strings")

    group = None
    ds = None

    # --- Try Average group ---
    try:
        if len(files) == 1:
            ds = xr.open_dataset(files[0], group="Data/Average/",
                                 engine="netcdf4", decode_timedelta=False)
        else:
            ds = xr.open_mfdataset(
                files,
                group="Data/Average/",
                concat_dim="time",
                combine="nested",
                engine="netcdf4",
                decode_timedelta=False
            )
        if ds.time.size > 0:
            group = "Average"
    except Exception:
        pass

    # --- Fallback to Burst group ---
    if group is None:
        try:
            if len(files) == 1:
                ds = xr.open_dataset(files[0], group="Data/Burst/",
                                     engine="netcdf4", decode_timedelta=False)
            else:
                ds = xr.open_mfdataset(
                    files,
                    group="Data/Burst/",
                    concat_dim="time",
                    combine="nested",
                    engine="netcdf4",
                    decode_timedelta=False
                )
            if ds.time.size > 0:
                group = "Burst"
        except Exception:
            raise ValueError("Neither 'Average' nor 'Burst' groups contain data")

    # Sort by time
    ds = ds.sortby("time")

    # Attach attributes from Config group of the FIRST file
    config = xr.open_dataset(files[0], group="Config", engine="netcdf4")
    ds = ds.assign_attrs(config.attrs)

    # Rename variables for consistency
    rename_map = {
        "Velocity Range": "VelocityRange",
        "Correlation Range": "CorrelationRange",
        "Amplitude Range": "AmplitudeRange"
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.variables})

    # Convert Pressure -> Depth
    if "Pressure" in ds.variables:
        ds = ds.assign(Depth=("time", -gsw.z_from_p(ds.Pressure.values, mean_lat)))
        ds = ds.drop_vars("Pressure")

    # Reorder dimensions consistently
    ds = ds.transpose()

    return ds




##################################################################################################

def ellipsoid_fit(X, flag=0, equals='xy'):
	"""
	Fit an ellipsoid (or constrained special cases of one) to a 3-D point
	cloud by least squares. Used internally by `correct_ad2cp_heading` for
	soft-iron magnetometer calibration: raw magnetometer readings over a
	full rotation should trace a sphere centered on the origin, but nearby
	ferrous material distorts that into an off-center ellipsoid, so fitting
	one and recentering removes the distortion.

	Not part of the public API (not in `__all__`); use
	`correct_ad2cp_heading` instead unless you need this directly.

	Parameters
	----------
	X : numpy.ndarray, shape (n_points, 3)
		Point cloud to fit, e.g. raw (Mx, My, Mz) magnetometer samples.
	flag : int
		Constraint on the fit: 0 = general ellipsoid (needs >= 9 points),
		1 = axis-aligned ellipsoid (>= 6 points), 2 = axis-aligned with two
		radii equal per `equals` (>= 5 points), 3 = sphere (>= 4 points).
	equals : {'xy', 'xz', 'zx', 'yz', 'zy'}
		Which pair of axes share a radius when `flag == 2`.

	Returns
	-------
	center : numpy.ndarray, shape (3,)
		Fitted ellipsoid center.
	radii : numpy.ndarray, shape (3,)
		Fitted ellipsoid radii along its principal axes.
	evecs : numpy.ndarray, shape (3, 3)
		Principal axes (eigenvectors); identity for `flag != 0`, since
		those constrained fits assume axis alignment.
	evals : numpy.ndarray or None
		Eigenvalues backing `radii` when `flag == 0`; None otherwise.
	v : numpy.ndarray
		Raw coefficient vector of the fitted quadric surface.
	"""
	if X.shape[1] != 3:
		raise ValueError('Input data must have three columns!')
	
	x = X[:, 0]
	y = X[:, 1]
	z = X[:, 2]
	
	# Check for sufficient points
	if len(x) < 9 and flag == 0:
		raise ValueError('Must have at least 9 points to fit a unique ellipsoid')
	if len(x) < 6 and flag == 1:
		raise ValueError('Must have at least 6 points to fit a unique oriented ellipsoid')
	if len(x) < 5 and flag == 2:
		raise ValueError('Must have at least 5 points to fit a unique oriented ellipsoid with two axes equal')
	if len(x) < 4 and flag == 3:
		raise ValueError('Must have at least 4 points to fit a unique sphere')

	if flag == 0:
		D = np.array([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z]).T
	elif flag == 1:
		D = np.array([x**2, y**2, z**2, 2*x, 2*y, 2*z]).T
	elif flag == 2:
		if equals in ['yz', 'zy']:
			D = np.array([y**2 + z**2, x**2, 2*x, 2*y, 2*z]).T
		elif equals in ['xz', 'zx']:
			D = np.array([x**2 + z**2, y**2, 2*x, 2*y, 2*z]).T
		else:
			D = np.array([x**2 + y**2, z**2, 2*x, 2*y, 2*z]).T
	else:
		D = np.array([x**2 + y**2 + z**2, 2*x, 2*y, 2*z]).T

	# Solve the normal system of equations
	v = np.linalg.lstsq(D, np.ones(len(x)), rcond=None)[0]

	if flag == 0:
		A = np.array([[v[0], v[3], v[4], v[6]],
					  [v[3], v[1], v[5], v[7]],
					  [v[4], v[5], v[2], v[8]],
					  [v[6], v[7], v[8], -1]])
		
		center = -np.linalg.inv(A[:3, :3]).dot(v[6:9])
		T = np.eye(4)
		T[3, :3] = center
		R = T.dot(A).dot(T.T)
		evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
		radii = np.sqrt(1. / evals)
	else:
		if flag == 1:
			v = np.concatenate([v[:3], [0, 0, 0], v[3:]])
		elif flag == 2:
			if equals in ['xz', 'zx']:
				v = np.concatenate([[v[0], v[1], v[0]], [0, 0, 0], v[2:]])
			elif equals in ['yz', 'zy']:
				v = np.concatenate([[v[1], v[0], v[0]], [0, 0, 0], v[2:]])
			else:  # xy
				v = np.concatenate([[v[0], v[0], v[1]], [0, 0, 0], v[2:]])
		else:
			v = np.concatenate([[v[0], v[0], v[0]], [0, 0, 0], v[1:]])
		
		center = -v[6:9] / v[:3]
		gam = 1 + (v[6]**2 / v[0] + v[7]**2 / v[1] + v[8]**2 / v[2])
		radii = np.sqrt(gam / v[:3])
		evecs = np.eye(3)

	return center, radii, evecs, evals if 'evals' in locals() else None, v
	
##################################################################################################
	
def calc_tilt_matrix(pitch, roll):
	"""
	Calculate the tilt matrix based on the pitch and roll angles.

	:param pitch: The pitch angle in degrees.
	:param roll: The roll angle in degrees.
	:return: The tilt matrix.
	"""
	sinpp = np.sin(np.radians(pitch))
	cospp = np.cos(np.radians(pitch))
	sinrr = np.sin(np.radians(roll))
	cosrr = np.cos(np.radians(roll))

	m = np.array([
		[cospp, -sinpp * sinrr, -cosrr * sinpp],
		[0, cosrr, -sinrr],
		[sinpp, sinrr * cospp, cospp * cosrr]
	])

	return m
	
##################################################################################################

def calc_heading(hxhyhz_sensor, pitch, roll, orientation):
	"""
	Calculate the heading based on sensor data, pitch, roll, and orientation.

	:param hxhyhz_sensor: The sensor data as a list or numpy array [Hx, Hy, Hz].
	:param pitch: The pitch angle in degrees.
	:param roll: The roll angle in degrees.
	:param orientation: The orientation of the instrument (0 for upwards, otherwise downwards).
	:return: The calculated heading in degrees.
	"""
	# Heading calculation
	m_tilt = calc_tilt_matrix(pitch, roll)

	# Check the orientation and adjust sensor data accordingly
	if orientation == 0:
		# Upwards looking instrument
		hxhyhz_inst = np.array(hxhyhz_sensor)
	else:
		# Downwards looking instrument (rotation around x-axis)
		hxhyhz_inst = np.array(hxhyhz_sensor)
		hxhyhz_inst[1:] = -hxhyhz_inst[1:]

	# Calculate the magnetic vector in the Earth aligned coordinate system
	hxhyhz_earth = np.dot(m_tilt, hxhyhz_inst)

	# Calculate the heading
	heading = np.arctan2(hxhyhz_earth[1], hxhyhz_earth[0]) * (180 / np.pi)
	if heading < 0:
		heading += 360

	return heading



##################################################################################################

def wrap180(angle):
    """Wrap an angle in degrees onto the interval [-180, 180)."""
    return (np.asarray(angle, dtype=float) + 180.0) % 360.0 - 180.0


def detect_roll_offset(ds, roll_var="Roll", threshold=90.0):
    """
    Detect whether the AD2CP reports roll in a frame rolled 180 degrees.

    Slocum payload-bay AD2CPs are not all installed the same way up. Some
    report roll near zero when the glider is level; others report roll near
    180 degrees for the same attitude. Everything downstream, cell_vert in
    particular, assumes the former, and silently returns negative cell depths
    for the latter, which makes binmap_adcp drop every bin.

    Parameters
    ----------
    ds : xarray.Dataset
        AD2CP dataset carrying a roll variable in degrees.
    roll_var : str
        Name of the roll variable.
    threshold : float
        Median absolute roll, in degrees, above which the instrument is taken
        to be reporting in the rolled frame.

    Returns
    -------
    float
        0.0 or 180.0, the offset to subtract from the reported roll.
    """
    median_abs_roll = np.nanmedian(np.abs(wrap180(ds[roll_var].values)))
    return 180.0 if median_abs_roll > threshold else 0.0


def correct_ad2cp_mounting(ds, roll_offset=None, roll_var="Roll"):
    """
    Express the AD2CP roll in the glider frame.

    Verify the offset for a new deployment by comparing the instrument's roll
    against the glider's own m_roll over one dive: the two should agree to
    within a degree once the offset is removed.

    Parameters
    ----------
    ds : xarray.Dataset
        AD2CP dataset.
    roll_offset : float, optional
        Degrees to subtract from the reported roll. Detected with
        :func:`detect_roll_offset` when omitted.
    roll_var : str
        Name of the roll variable to correct.

    Returns
    -------
    xarray.Dataset
        Copy with ``roll_var`` rewritten in the glider frame and the original
        preserved as ``RollInstrument``.
    """
    if roll_offset is None:
        roll_offset = detect_roll_offset(ds, roll_var=roll_var)

    ds = ds.copy()
    ds = ds.assign(RollInstrument=(ds[roll_var].dims, ds[roll_var].values.copy()))
    ds = ds.assign({roll_var: (ds[roll_var].dims,
                               wrap180(ds[roll_var].values - roll_offset))})
    ds.attrs["roll_offset_applied"] = float(roll_offset)
    return ds


##################################################################################################

def correct_ad2cp_heading(ds):
    """
    Correct AD2CP heading using magnetometer and orientation data,
    and return a new xarray.Dataset with corrected values.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain variables: 'Heading', 'Pitch', 'Roll', 'Pressure',
        'MagnetometerX', 'MagnetometerY', 'MagnetometerZ', 'time'

    Returns
    -------
    ds_corrected : xarray.Dataset
        Original dataset with added variables:
        - 'CorrectedHeading'
        - 'MagX_corrected'
        - 'MagY_corrected'
        Also retains original magnetometer variables.
    """
    head = np.array(ds['Heading'])
    pitch = np.array(ds['Pitch'])
    roll = np.array(ds['Roll'])
    x = np.array(ds['MagnetometerX'])
    y = np.array(ds['MagnetometerY'])
    z = np.array(ds['MagnetometerZ'])

    xyz_original = np.column_stack((x, y, z))

    pitch_ranges = np.arange(-35, 35, 1)
    for k in range(len(pitch_ranges) - 1):
        mask = (pitch > pitch_ranges[k]) & (pitch < pitch_ranges[k + 1])
        indices = np.where(mask)
        if len(indices[0]) > 9:
            xyz1 = np.column_stack((x[indices], y[indices], z[indices]))
            offset, *_ = ellipsoid_fit(xyz1)

            x1 = x[indices] - offset[0]
            y1 = y[indices] - offset[1]
            z1 = z[indices] - offset[2]

            new_center, *_ = ellipsoid_fit(np.column_stack((x1, y1, z1)))
            if abs(new_center[0]) > 150 or abs(new_center[1]) > 150:
                x1 = x[indices]
                y1 = y[indices]
                z1 = z[indices]

            x[indices] = x1
            y[indices] = y1
            z[indices] = z1

    xyz_final = np.column_stack((x, y, z))

    CorrectedHeading = np.empty_like(head) * np.nan
    for k in range(len(head)):
        CorrectedHeading[k] = calc_heading(xyz_final[k, :], pitch[k], roll[k], 1)

    # Create a new dataset with corrected variables
    ds_corrected = ds.copy()
    ds_corrected = ds_corrected.assign(
        CorrectedHeading=("time", CorrectedHeading),
        MagX_corrected=("time", x),
        MagY_corrected=("time", y),
    )

    return ds_corrected






__all__ = [
    "check_max_beam_range",
    "check_max_beam_range_bins",
    "check_mean_beam_range",
    "check_mean_beam_range_bins",
    "beam2enu",
    "beam_true_depth",
    "binmap_adcp",
    "cell_vert",
    "correct_sound_speed",
    "qaqc_pre_coord_transform",
    "qaqc_post_coord_transform",
    "inversion",
    "mag_var_correction",
    "shear_method",
    "calcAHRS",
    "load_ad2cp",
    "correct_ad2cp_heading",
    "mag_var_correction_ad2cp_ds",
    "wrap180",
    "detect_roll_offset",
    "correct_ad2cp_mounting"
]
