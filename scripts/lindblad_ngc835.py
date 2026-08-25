import os
import argparse
import configparser
import json
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
import astropy.units as u
from astropy.visualization.wcsaxes import add_scalebar
from matplotlib.patches import Rectangle, Ellipse, Annulus
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-p', '--ssp', dest='ssp_path', \
                        help='Path to SSP data.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders may be created there.')
    args = parser.parse_args()
    return args

def scalebar_angle(d, bar_len = 10 * u.kpc):
    return (bar_len / d).to(u.deg, equivalencies=u.dimensionless_angles())

def get_z(num):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get('ngc'+num, "z"))
    return z

def tremaine_weinberg(
    starvel,
    starvel_err,
    vband,
    i,
    pa,
    cx,
    cy,
    slit_width=1,
    n_slits=None,
    extra_mask=None,
    min_pixels_per_slit=5,
    pixel_to_arcsec=0.2,
    pixel_to_kpc=0.3,
    flux_threshold=None
):
    """
    Compute the bar/spiral pattern speed with the Tremaine-Weinberg method.

    Parameters
    ----------
    starvel : 2D ndarray
        Stellar line-of-sight velocity map [km/s].  NaNs mark missing data.
    starvel_err : 2D ndarray
        Uncertainty map for starvel [km/s].  NaNs or non-positive values are masked automatically.
    vband : 2D ndarray
        V-band (or any continuum tracer) surface-brightness map.
        NaNs or non-positive values are masked automatically.
    i : float
        Galaxy inclination [degrees].  Must be in (0, 90).
    pa : float
        Position angle of the **receding** major axis [degrees],
        measured East of North (same convention as kinematic PAs from pPXF /
        Kinemetry).  The code rotates coordinates so this axis becomes the
        positive x-direction.
    cx, cy : float
        Galaxy centre in pixel coordinates (0-based, x = column, y = row).
    slit_width : float, optional
        Width of each pseudo-slit in pixels (default 1).
    n_slits : int or None, optional
        Number of pseudo-slits.  If None the slits span the full extent of
        valid data with the given ``slit_width``.
    extra_mask : 2D bool ndarray or None, optional
        Additional bad-pixel mask (True = bad).
    min_pixels_per_slit : int, optional
        Slits with fewer valid pixels than this are discarded (default 5).
    flux_threshold : float or None, optional
        Pixels with vband < flux_threshold are masked.  If None, defaults to
        1% of the 99th-percentile flux.

    Returns
    -------
    result : dict
        Dictionary with keys:
        ``TW_X``, ``TW_V``, ``TW_EV``, ``slit_positions``.
    """

    starvel = np.asarray(starvel, dtype=float)
    starvel_err = np.asarray(starvel_err, dtype=float)
    vband   = np.asarray(vband,   dtype=float)

    if starvel.shape != vband.shape or starvel.shape != starvel_err.shape:
        raise ValueError("starvel, starvel_err and vband must have the same shape.")
    if not (0 < i < 90):
        raise ValueError("Inclination i must be strictly between 0 and 90 degrees.")

    ny, nx = starvel.shape

    # ── 1. Build centred coordinate grids ────────────────────────────────────
    xarr = np.arange(nx) - cx          # columns → x
    yarr = np.arange(ny) - cy          # rows    → y
    X_img, Y_img = np.meshgrid(xarr, yarr)

    # ── 2. Rotate to galaxy frame (major axis → x_rot) ───────────────────────
    # pa is measured North-to-East; in image coords North is usually +y.
    # Rotation by pa anti-clockwise brings major axis to x_rot.
    pa_rad = np.deg2rad(pa-90)
    X_rot =  X_img * np.cos(pa_rad) + Y_img * np.sin(pa_rad)
    Y_rot = -X_img * np.sin(pa_rad) + Y_img * np.cos(pa_rad)
    # X_rot → along major axis (signed, receding side positive)
    # Y_rot → along minor axis (slit separation axis)

    # Build combined bad-pixel mask
    if flux_threshold is None:
        flux_threshold = 0.01 * np.nanpercentile(vband, 99)

    mask = (
        np.isnan(starvel)
        | np.isnan(vband)
        | (vband <= flux_threshold)
    )
    if extra_mask is not None:
        mask |= np.asarray(extra_mask, dtype=bool)

    vel_clean  = np.where(mask, np.nan, starvel)
    vel_err_clean = np.where(mask, np.nan, starvel_err)
    flux_clean = np.where(mask, np.nan, vband)

    # Define pseudo-slit positions along Y_rot
    valid_y = Y_rot[~mask]
    if valid_y.size == 0:
        raise ValueError("No valid pixels after masking.")

    y_min = valid_y.min()
    y_max = valid_y.max()

    print(valid_y, y_min, y_max)
    if n_slits is not None:
        slit_centers = np.linspace(y_min + slit_width / 2,
                                   y_max - slit_width / 2,
                                   int(n_slits))
    else:
        slit_centers = np.arange(
            y_min + slit_width / 2,
            y_max + slit_width / 2,
            slit_width
        )

    # Compute luminosity-weighted <X> and <V> per slit
    TW_X, TW_V, TW_EV, slit_pos = [], [], [], []
    plotidx = 0
    for yc in slit_centers:
        in_slit = (
            (Y_rot >= yc - slit_width / 2)
            & (Y_rot <  yc + slit_width / 2)
            & (~mask)
        )
        if in_slit.sum() < min_pixels_per_slit:
            continue

        f = flux_clean[in_slit]
        v = vel_clean[in_slit]
        ev = vel_err_clean[in_slit]
        x = X_rot[in_slit]

        total_flux = np.nansum(f)
        if total_flux <= 0:
            continue

        TW_X.append(np.nansum(f * x) / total_flux)
        TW_V.append(np.nansum(f * v) / total_flux)
        TW_EV.append(np.sqrt(np.nansum((f * ev)**2)) / total_flux)
        slit_pos.append(yc)

    TW_X     = np.array(TW_X)
    TW_V     = np.array(TW_V)
    TW_EV   = np.array(TW_EV)
    slit_pos = np.array(slit_pos)

    if len(TW_X) < 3:
        raise RuntimeError(
            f"Only {len(TW_X)} valid slits found — too few for a reliable fit. "
            "Try reducing slit_width or min_pixels_per_slit."
        )

    result = dict(
        TW_X         = TW_X,
        TW_V         = TW_V,
        TW_EV        = TW_EV,
        slit_positions = slit_pos,
    )

    return result

def rotation_curve(vel, flux, wcs, cx, cy, width, height, pa, arc_pix, out_dir, num):
    h_img, w_img = vel.shape
    y, x = np.mgrid[:h_img, :w_img]
    dx = x - cx
    dy = y - cy
    theta = np.deg2rad(90+pa)
    # Along-slit coordinate (radius)
    s = dx * np.cos(theta) + dy * np.sin(theta)
    # Across-slit coordinate
    t = -dx * np.sin(theta) + dy * np.cos(theta)
    # Mask to select pixels within the slit
    mask = (np.abs(s) <= height/2) & (np.abs(t) <= width/2)

    plot_slit(vel, flux, mask, cx, cy, width, height, pa, wcs, out_dir, num)

    r = s[mask]                  # radius along slit
    v = vel[mask]

    bins = np.linspace(-height/2, height/2, int(height/4))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    profile = []
    for i in range(len(bins) - 1):
        sel = (r >= bins[i]) & (r < bins[i+1])
        if np.any(sel):
            profile.append(np.mean(v[sel]))
        else:
            profile.append(np.nan)

    profile = np.array(profile)
    bin_centers = bin_centers * arc_pix.value # from pixel to arcsec

    # Center the profile on the systemic velocity
    cent = np.where(np.abs(bin_centers) == np.min(np.abs(bin_centers)))[0][0]
    vcent = profile[cent]
    print(f'Velocity at {bin_centers[cent] * arc_pix.value} arcsec = {profile[cent]} km/s')
    profile = profile - vcent # central value in 0

    return bin_centers, profile, vcent

def plot_slit(vel, flux, mask, cx, cy, width, height, pa, wcs, out_dir, num):
    z = get_z(num)
    d = cosmo.luminosity_distance(z).to(u.Mpc).value
    rect = Rectangle((cx-width/2, cy-height/2),
                width=width, height=height,
                linewidth=2, angle=pa, rotation_point='center')

    fig, ax = plt.subplots(ncols=3, figsize=(14,6), subplot_kw=dict(projection=wcs))
    plt.subplots_adjust(wspace=0.02)
    ax[0].imshow(flux, cmap='cividis', norm='log')
    ax[0].scatter(cx, cy, marker='+', color='k', s=50, linewidths=1)
    add_scalebar(ax[0], scalebar_angle(d * u.Mpc, 5*u.kpc), label="5 kpc", color="k", label_top=True, corner="top left")
    im = ax[1].imshow(vel, cmap='coolwarm')
    # plt.colorbar(im, ax=ax[1], shrink=0.53, pad=0.02)
    ax[1].add_patch(rect)
    ax[1].scatter(cx, cy, marker='+', color='k', s=50, linewidths=1)
    ax[2].imshow(mask)
    for a in ax:
        a.set_xlabel('RA', fontsize=16)
        a.tick_params(labelsize=14)
        if a != ax[0]:
            a.coords[0].set_ticklabel_visible(False)  # hide tick labels
            a.coords[0].set_axislabel('')             # remove axis label
        else:
            a.set_ylabel('Dec', fontsize=16)
    
    plt.savefig(os.path.join(out_dir, f"NGC{num}_slit.pdf"), dpi=300, bbox_inches='tight')

def angular_velocity(bin_centers, profile, kpc_arc, out_dir, num):
    """
    No need to correct for inclination because it is already taken into account.
    In this case we only take the negative side of the profile, which is the receding side, to avoid contamination (bubble) from the approaching side.
    Saves the image og the angular velocity and epicyclic frequency profiles.

    Parameters
    ----------
    bin_centers : 1D array
        Radial positions along the slit (in arcsec).
    profile : 1D array
        Line-of-sight velocity profile along the slit (in km/s), centered on the systemic velocity.
    kpc_arc : float
        Conversion factor from arcsec to kpc at the distance of the galaxy (in kpc/arcsec).
    
    Returns
    -------
    rad_kpc : 1D array
        Radial positions along the slit in kpc.
    omega : 1D array
        Angular velocity profile (in km/s/kpc).
    kappa : 1D array
        Epicyclic frequency profile (in km/s/kpc).    
    """
    maskrad = bin_centers < -2
    rad = np.abs(bin_centers[maskrad])[::-1]
    v = np.abs(profile[maskrad])[::-1]

    rad_kpc = rad * kpc_arc

    omega = v / rad_kpc # km/s/kpc
    # smooth
    omega_s = savgol_filter(omega, 7, 3)
    # derivative
    domega_dr = np.gradient(omega_s, rad_kpc)
    # kappa
    kappa2 = 4*omega_s**2 + 2*rad_kpc*omega_s*domega_dr
    kappa = np.sqrt(kappa2)

    plot_omega_rad(rad_kpc, omega, kappa, out_dir, num)

    return rad_kpc, omega, kappa

def plot_omega_rad(rad_kpc, omega, kappa, out_dir, num, omega_p=None, omega_p_err=None, reso=None, cr=None, reso_unc=None, cr_unc=None):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(rad_kpc, omega, 'k', label=r'$\Omega$')
    ax.plot(rad_kpc, omega - kappa/2, '--', color='k', label='$\\Omega\\pm \\kappa/2$')
    ax.plot(rad_kpc, omega + kappa/2, '--', color='k')

    if omega_p is not None and omega_p_err is not None:
        ax.axhline(omega_p, color='C0', linestyle='-.', label=f'$\\Omega_P$')
        ax.fill_between(rad_kpc, omega_p-omega_p_err, omega_p+omega_p_err, alpha=0.15)

    if reso is not None and reso_unc is not None:
        if len(reso_unc)%2 != 0:
            raise ValueError("reso_unc should contain pairs of low and high values for each resonance.")
        omax = np.max(omega + kappa/2)
        omax = np.round((omax+20)/100) * 100
        for j,re in enumerate(reso):
            if j == 0:
                ax.axvline(re, color='g', linestyle=':', label=f'ILR')
            else:
                ax.axvline(re, color='g', linestyle=':')
        for j in range(0,len(reso_unc),2):
            ax.fill_betweenx(np.arange(0, omax), reso_unc[j], reso_unc[j+1], color='g', alpha=0.15)
    if cr is not None and cr_unc is not None:
        if len(cr_unc)%2 != 0:
            raise ValueError("cr_unc should contain pairs of low and high values for CR.")
        omax = np.max(omega + kappa/2)
        omax = np.round((omax+20)/100) * 100
        ax.axvline(cr, color='orange', linestyle=':', label=f'CR')
        for j in range(0,len(cr_unc),2):
            ax.fill_betweenx(np.arange(0, omax), cr_unc[j], cr_unc[j+1], color='orange', alpha=0.15)

    ax.legend(fontsize=12)
    ax.set_xlabel('Radius (kpc)', fontsize=16)
    ax.set_ylabel('$\\Omega$ (km/s/kpc)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    if omega_p is not None or reso is not None or cr is not None:
        ax.set_ylim(0, np.max(omega + kappa/2)+40)
        plt.savefig(os.path.join(out_dir, f"NGC{num}_omega_kappa_resonances.pdf"), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(out_dir, f"NGC{num}_omega_kappa.pdf"), dpi=300, bbox_inches='tight')


def fit_omega_p(x, v, ev, inc, arc_pix, kpc_arc):
    weights = 1.0 / ev         # polyfit expects 1/error, not 1/error**2, does transform later

    coeffs, cov = np.polyfit(x, v, deg=1, w=weights, cov=True)
    slope, intercept = coeffs
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    i_rad       = np.deg2rad(inc)
    omega_p_pix     = slope     / np.sin(i_rad)
    omega_p_pix_err = slope_err / np.sin(i_rad)
    omega_p = omega_p_pix/(arc_pix * kpc_arc).value # km/s/kpc
    omega_p_err = omega_p_pix_err/(arc_pix * kpc_arc).value # km/s/kpc
    print(f"Pattern speed  Ω_p: {omega_p} +- {omega_p_err}")

    return omega_p, omega_p_err, slope, intercept, cov

def plot_tw(result, slope, intercept, cov, out_dir, num, window=False):
    c = np.linspace(np.min(result['slit_positions']), np.max(result['slit_positions']), len(result['TW_X']))
    norm = colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = cm.plasma

    fig, ax = plt.subplots(figsize=(7,5))
    for xi, yi, ei, ci in zip(result['TW_X'], result['TW_V'], result['TW_EV'], c):
        col = cmap(norm(ci))
        ax.errorbar(
            xi, yi,
            yerr=ei,
            fmt='o',
            color=col,
            ecolor=col,
            elinewidth=1,
            # markeredgecolor='black',  # optional for contrast
            markerfacecolor=col
        )

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Slit position (pix)', fontsize=14)
    ax.set_ylim(-100, 280)

    # Fit line
    xfit = np.linspace(np.min(result['TW_X']), np.max(result['TW_X']), 200)
    yfit = slope * xfit + intercept
    var_yfit = (xfit**2 * cov[0, 0] + cov[1, 1] + 2 * xfit * cov[0, 1])
    yfit_err = np.sqrt(var_yfit)
    ax.plot(xfit, yfit, 'r-', lw=2)

    ax.fill_between(xfit, yfit - yfit_err, yfit + yfit_err, color='red', alpha=0.3)
    ax.set_xlabel('$\\langle X\\rangle$ (pix)', fontsize=16)
    ax.set_ylabel('$\\langle V \\rangle$ (km/s)', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    if window:
        axins = inset_axes(ax, width="40%", height="40%", bbox_to_anchor=(0.30, -0.15, 0.85, 0.75),
                        bbox_transform=ax.transAxes, loc='center')

        for xi, yi, ei, ci in zip(result['TW_X'], result['TW_V'], result['TW_EV'], c):
            col = cmap(norm(ci))

            axins.errorbar(
                xi, yi,
                yerr=ei,
                fmt='o',
                color=col,
                ecolor=col,
                elinewidth=1,
                # markeredgecolor='black',  # optional for contrast
                markerfacecolor=col
            )

        axins.plot(xfit, yfit, 'r-', lw=2)
        axins.fill_between(xfit, yfit - yfit_err, yfit + yfit_err, color='red', alpha=0.3)
        # Define the zoomed region (adjust as needed)
        x1, x2 = -10, 10
        y1, y2 = -50, 130
        axins.set_ylim(y1, y2)
        axins.set_xlim(x1,x2)
        axins.yaxis.tick_right()

        ax.annotate("", xytext=(37, 40), xy=(10, 55),
                    arrowprops=dict(arrowstyle="-"))
        ax.annotate("", xytext=(37, -75), xy=(0, 10),
                    arrowprops=dict(arrowstyle="-"))
    
    plt.savefig(os.path.join(out_dir, f"NGC{num}_TW.pdf"), dpi=300, bbox_inches='tight')

def get_resonances(omega, kappa, rad_kpc, omega_p, omega_p_err):
    # Exact position of resonances
    reso = []
    cr = None
    for j, val in enumerate(omega - kappa/2):
        if j == 0: continue
        if val > omega_p and (omega - kappa/2)[j-1] < omega_p:
            m = (val-(omega - kappa/2)[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = val - m * rad_kpc[j]
            reso.append((omega_p - n)/m)
        elif val < omega_p and (omega - kappa/2)[j-1] > omega_p:
            m = (val-(omega - kappa/2)[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = val - m * rad_kpc[j]
            reso.append((omega_p - n)/m)
        
        if omega[j] < omega_p and omega[j-1] > omega_p:
            m = (omega[j]-omega[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = omega[j] - m * rad_kpc[j]
            cr = (omega_p - n)/m
    print('CR: ', cr)
    print('iILR: ', reso[0])
    print('oILR: ', reso[1])

    # Uncertainty of resonances
    reso_low, reso_high = [], []
    cr1, cr2 = None, None
    for j, val in enumerate(omega - kappa/2):
        if j == 0: continue
        if val > omega_p-omega_p_err and (omega - kappa/2)[j-1] < omega_p-omega_p_err:
            m = (val-(omega - kappa/2)[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = val - m * rad_kpc[j]
            reso_low.append((omega_p-omega_p_err - n)/m)
        if val > omega_p+omega_p_err and (omega - kappa/2)[j-1] < omega_p+omega_p_err:
            m = (val-(omega - kappa/2)[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = val - m * rad_kpc[j]
            reso_high.append((omega_p+omega_p_err - n)/m)
        if val < omega_p-omega_p_err and (omega - kappa/2)[j-1] > omega_p-omega_p_err:
            m = (val-(omega - kappa/2)[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = val - m * rad_kpc[j]
            reso_low.append((omega_p-omega_p_err - n)/m)
        if val < omega_p+omega_p_err and (omega - kappa/2)[j-1] > omega_p+omega_p_err:
            m = (val-(omega - kappa/2)[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = val - m * rad_kpc[j]
            reso_high.append((omega_p+omega_p_err - n)/m)

        if omega[j] < omega_p+omega_p_err and omega[j-1] > omega_p+omega_p_err:
            m = (omega[j]-omega[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = omega[j] - m * rad_kpc[j]
            cr1 = (omega_p+omega_p_err - n)/m
        if omega[j] < omega_p-omega_p_err and omega[j-1] > omega_p-omega_p_err:
            m = (omega[j]-omega[j-1])/(rad_kpc[j]-rad_kpc[j-1])
            n = omega[j] - m * rad_kpc[j]
            cr2 = (omega_p-omega_p_err - n)/m
            
    print('CR low: ', cr1)
    print('CR high: ', cr2)
    print('LR low: ', reso_low)
    print('LR high: ', reso_high)
    reso_high = np.array(reso_high)
    reso_low = np.array(reso_low)

    return reso, cr, reso_low, reso_high, [cr1, cr2]

def plot_continuum_resonances(vband, wcs, reso_unc, cr_unc, inc, pa, cx, cy, kpc_arc, arc_pix, out_dir, num):
    fig, ax = plt.subplots(figsize=(8,6), subplot_kw=dict(projection=wcs))
    im = ax.imshow(vband, norm='log', cmap='nipy_spectral', vmin=3e-3, vmax=0.1)
    # ax.scatter(cx, cy, marker='+', color='k', s=65, linewidths=1)
    ax.scatter(250, 271, marker='+', color='w', s=120, linewidths=2) #bubble
    cbar = plt.colorbar(im, ax=ax, shrink=1, pad=0.02);
    cbar.set_label(f"Flux ($10^{{-16}}$ erg/s/cm$^2$)", rotation=270, fontsize=14, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    for j in range(0, len(reso_unc), 2):
        wi = reso_unc[j]/kpc_arc.value/arc_pix.value
        hi = reso_unc[j]/kpc_arc.value/arc_pix.value/np.cos(np.deg2rad(inc))
        wo = reso_unc[j+1]/kpc_arc.value/arc_pix.value
        ho = reso_unc[j+1]/kpc_arc.value/arc_pix.value/np.cos(np.deg2rad(inc))
        ann = Annulus((cx, cy), (wo,ho), wo-wi, angle=pa, edgecolor='k', facecolor='none', alpha=1, linestyle='dashed')
        ax.add_patch(ann)

    wo = cr_unc[1]/kpc_arc.value/arc_pix.value
    ho = cr_unc[1]/kpc_arc.value/arc_pix.value/np.cos(np.deg2rad(inc))
    wi = cr_unc[0]/kpc_arc.value/arc_pix.value
    hi = cr_unc[0]/kpc_arc.value/arc_pix.value/np.cos(np.deg2rad(inc))

    ann = Annulus((cx, cy), (wo,ho), wo-wi, angle=pa, edgecolor='w', facecolor='none', alpha=1, linestyle='dashed')
    ax.add_patch(ann);

    ax.set_xlabel('RA', fontsize=16)
    ax.set_ylabel('Dec', fontsize=16)
    ax.tick_params(labelsize=14)

    plt.savefig(os.path.join(out_dir, f"NGC{num}_continuum_resonances.pdf"), dpi=300, bbox_inches='tight')

def main():
    args = get_args()
    ssp_path = args.ssp_path
    out_dir = args.out_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    num = ssp_path.lower().split('ngc')[-1][:3]
    z = get_z(num)
    d = cosmo.luminosity_distance(z).to(u.Mpc).value
    arc_pix = 0.2 * u.arcsec/u.pix
    kpc_arc = cosmo.kpc_proper_per_arcmin(z).to('kpc/arcsec')

    with fits.open(f'../output/ngc{num}/NGC{num}_vopt_Ha_masked.fits') as hdul:
        vel = hdul[0].data
        vel_err = hdul[1].data
        vel_header = hdul[0].header
        vel_wcs = WCS(vel_header)

    with fits.open(f'../output/ngc{num}/NGC{num}_flux_Ha_masked.fits') as hdul:
        flux = hdul[0].data
        flux_err = hdul[1].data
        fluxheader = hdul[0].header

    with fits.open(f'../data/NGC835.SSP.cube.fits') as hdul:
        vband = hdul[0].data[0]
        starvel = hdul[0].data[13]
        starvel_err = hdul[0].data[14]
        starvel[starvel == 0] = np.nan

    cx, cy = 244, 176
    width = 10 # pixels
    height = (50 * u.arcsec / arc_pix).value # pixels
    pa = 4.6 # * u.deg
    inc = 34 # * u.deg

    bin_centers, profile, vcent = rotation_curve(vel, flux, vel_wcs, cx, cy, width, height, pa, arc_pix, out_dir, num)
    rad_kpc, omega, kappa = angular_velocity(bin_centers, profile, kpc_arc.value, out_dir, num)

    # Do with stellar velocity, more reliable for TW method
    tw_result = tremaine_weinberg(
                starvel    = starvel-vcent,
                starvel_err = starvel_err,
                vband      = vband,
                i          = inc,
                pa         = pa+180,
                cx         = cx,
                cy         = cy,
                slit_width = 5,
                pixel_to_arcsec = arc_pix.value,
                pixel_to_kpc = (arc_pix * kpc_arc).value
            )
    
    omega_p, omega_p_err, slope, intercept, cov = fit_omega_p(tw_result['TW_X'], tw_result['TW_V'], tw_result['TW_EV'], inc, arc_pix, kpc_arc)

    plot_tw(tw_result, slope, intercept, cov, out_dir, num, window=False)

    reso, cr, reso_low, reso_high, cr_unc = get_resonances(omega, kappa, rad_kpc, omega_p, omega_p_err)
    if len(reso_high) == 0:
        reso_unc = [reso_low[0], reso_low[1]]
    else:
        reso_unc = [reso_low[0], reso_high[0], reso_low[1], reso_high[1]]

    plot_omega_rad(rad_kpc, omega, kappa, out_dir, num, omega_p=omega_p, omega_p_err=omega_p_err, reso=reso[:2], cr=cr, reso_unc=reso_unc, cr_unc=cr_unc)

    plot_continuum_resonances(vband, vel_wcs, reso_unc, cr_unc, inc, pa, cx, cy, kpc_arc, arc_pix, out_dir, num)

if __name__ == "__main__":
    main()