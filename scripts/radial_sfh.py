import os
import argparse
import configparser
import json
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.aperture import EllipticalAperture

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-p', '--ssp', dest='ssp_path', \
                        help='Path to SSP data.')
    parser.add_argument('-f', '--sfh', dest='sfh_path', \
                        help='Path to SSP data.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders may be created there.')
    args = parser.parse_args()
    return args

def flux_to_lum(flux, eflux, dist):
    """
    Converts flux (in erg/cm**2/s) to luminosity (in erg/s) given a distance (in Mpc)
    
    Parameters
    ----------
        flux : float or np.ma.MaskedArray
            Flux in erg/cm**2/s
        eflux : float or np.ma.MaskedArray
            Error in flux in erg/cm**2/s
        dist : float
            Distance to the galaxy in Mpc
    Returns
    -------
        lum : float or np.ma.MaskedArray
            Luminosity in erg/s
        elum : float or np.ma.MaskedArray
            Error in luminosity in erg/s
    """

    flux *= u.erg / u.cm**2 / u.s
    eflux *= u.erg / u.cm**2 / u.s
    dist *= u.Mpc
    lum, elum = 4*np.pi*dist.to(u.cm)**2*flux, 4*np.pi*dist.to(u.cm)**2*eflux

    if isinstance(flux, np.ma.MaskedArray):
        lum, elum = np.ma.MaskedArray(lum, mask=flux.mask), np.ma.MaskedArray(elum, mask=eflux.mask)

    return lum, elum

def get_z(num):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get('ngc'+num, "z"))
    return z

def read_data(ssp_path, sfh_path):
    # Read SSP data
    with fits.open(ssp_path) as hdul:
        ssp_header = hdul[0].header
        un = 1
        ssp_data = hdul[0].data
        if hdul[0].header['UNITS_3'] == ' 10^-16 erg':
            un = 1e-16
    ssp_data[3] = ssp_data[3] * un
    ssp_data[4] = ssp_data[4] * un
    # Read SFH data
    with fits.open(sfh_path) as hdul:
        sfh_data = hdul[0].data
        sfh_header = hdul[0].header
    return ssp_data, ssp_header, sfh_data, sfh_header

def get_ages(sfh_header):
    age = []
    for i in sfh_header:
        if 'DESC' in i:
            if 'age-met' in sfh_header[i]:
                a = sfh_header[i].split('-')
                if float(a[1].split(' ')[1]) in age:
                    continue
                age.append(float(a[1].split(' ')[1]))
    age = np.array(age)*u.Gyr
    return age

def sum_fractions(sfh_data, nbins):
    """
    Sum the fractions of the metallicity bins for each age bin. Unsorted.
    """
    _, ny, nx = sfh_data.shape
    sfh_metsum = np.zeros((nbins, ny, nx))
    for i in range(nbins):
        sfh_metsum[i, :, :] = np.sum(sfh_data[i*4:(i+1)*4, :, :], axis=0)
    return sfh_metsum

def calc_mass(ssp_data, fractions, nbins, z, mask=None):
    lum, elum = flux_to_lum(ssp_data[3], ssp_data[4], cosmo.luminosity_distance(z).to(u.Mpc).value)
    mass = np.zeros(nbins)
    emass = np.zeros(nbins)
    mass2light = 10**ssp_data[17] # 10**
    mass2light[mass2light < 0] = 0 # np.nan or 0
    if mask is None:
        mask = np.ones(lum.shape, dtype=bool)
    for i in range(nbins):
        mass[i] = np.nansum(fractions[i] * lum.to(u.L_sun).value * mass2light * mask)
        emass[i] = np.nansum(fractions[i] * elum.to(u.L_sun).value * mass2light * mask)
    return lum, elum, mass, emass

def get_bins(ages):
    """
    Return parameters needed for plotting the SFH histogram with ages in Myr.

    Parameters
    ----------
    ages : np.ndarray
        Array of age bin centers in Gyr.
    """
    # log10 of bin centers
    logt = np.log10(ages.value)
    # allocate edges
    edges = np.zeros(len(ages) + 1)
    # interior edges = midpoints in log space
    edges[1:-1] = 10 ** ((logt[1:] + logt[:-1]) / 2)
    # extrapolate first and last edges
    edges[0] = ages.value[0] / (10 ** ((logt[1] - logt[0]) / 2))
    edges[-1] = ages.value[-1] * (10 ** ((logt[-1] - logt[-2]) / 2))
    # bin widths
    dt = edges[1:] - edges[:-1]
    dt = dt * u.Gyr
    dt_yr = dt.to(u.yr).value
    # Compute bin centers
    bin_centers = (edges[:-1] + edges[1:]) / 2 * 1e3  # in Myr
    return bin_centers, edges*1e3, dt, dt_yr

def savefile(text, out_dir, num, sstr, estr):
    if not os.path.exists(f'{out_dir}/SFR_ngc{num}.txt'):
        with open(f'{out_dir}/SFR_ngc{num}.txt', 'w') as f:
            f.write(text)
        return
    with open(f'{out_dir}/SFR_ngc{num}.txt', 'r') as f:
        lines = f.readlines()
    
    found = False
    for i, line in enumerate(lines):
        if line.startswith(sstr):
            start = i
            found = True
        else:
            if found and line.startswith(estr):
                end = i + 1 # next newline
                break
    if found:
        lines[start:end+1] = text.splitlines(keepends=True)
    else:
        lines.append(text)
    
    with open(f'{out_dir}/SFR_ngc{num}.txt', 'w') as f:
        f.writelines(lines)

def make_ellipses(image, out_dir, num, radii=None, thin=False):
    """
    Create ring masks

    Parameters
    ----------
    image : array-like
        Image for which are the masks. Usually (y,x).
    radii : list, optional
        Values of the effective radii to make the masks, in pixels. If not given it will take from 1/10 to 4 HLR.

    Returns
    ---------
    center : tuple
        Center of ellipses.
    masks : list
        List of masks for ellipses.
    """
    # Estimate image center from brightest pixel
    image = image[0]
    center = np.unravel_index(np.argmax(gaussian_filter(image, sigma=5)), image.shape)
    center = (center[1], center[0]) # (x, y)
    # Create guess ellipse
    geometry = EllipseGeometry(x0=center[0], y0=center[1], sma=100, eps=0.5,
                           pa=45.0 * np.pi / 180.0)
    # Make fit
    ellipse = Ellipse(image, geometry)
    isolist = ellipse.fit_image()
    iso = isolist.get_closest(125)
    eps = iso.eps
    pa = iso.pa
    # Calculate effective radius HLR
    flux_values  = isolist.tflux_e # total flux enclosed within each isophote
    total_flux   = flux_values[-1]
    interp = interp1d(flux_values, isolist.sma)
    r_e = interp(total_flux / 2.0) # semi-major axis
    hlr = np.sqrt(r_e**2*(1-eps)) # effective radius
    if radii is None:
        smaja = np.arange(r_e/10, 4*r_e+r_e/15, r_e/10)
        radii = np.sqrt(smaja**2*(1-eps))
        smina = radii**2/smaja
        radii = radii / hlr # radii as a function of HLR
        radii = np.round(radii,1)
        radname = '_HLR'
    else:
        smaja = np.sqrt(radii**2/(1-eps))
        smina = radii**2/smaja
        radname = ''

    smaja = np.hstack([smaja, r_e])
    smina = np.hstack([smina, r_e * (1 - eps)])
    
    # Save fit parameters
    spa = pa * 180 / np.pi + 90
    if spa > 180:
        spa -= 180
    text = f"Ellipse fit parameters for NGC{num}\n(Values in pixels unless specified)\nPA: {spa:.2f} deg\nEccentricity: {eps:.3f}\nHLR Effective radius: {hlr:.2f}\nSemi-major axis: {r_e:.2f}\nSemi-minor axis: {r_e * (1 - eps):.2f}\nCenter (x,y): {center[0]}, {center[1]}\n\n"
    savefile(text, out_dir, num, "Ellipse fit parameters", "Center")
    # Plot ellipses
    plot_ellipses(image, center, smaja, smina, pa, out_dir, num, radname)
    # Create masks
    masks = []
    for k in range(len(smaja)):
        mask = np.zeros(image.shape, dtype=bool)
        ys, xs = np.indices(image.shape)
        dx = xs - center[0]
        dy = ys - center[1]

        # u =  dx * np.sin(pa) - dy * np.cos(pa)  # along major axis
        # v =  dx * np.cos(pa) + dy * np.sin(pa)  # along minor axis
        ##### different PA convention for this equation.
        u =  dx * np.cos(pa) + dy * np.sin(pa)  # along major axis
        v =  -dx * np.sin(pa) + dy * np.cos(pa)  # along minor axis
        # condition to see if it is effective radius HLR
        if k == len(smaja) -1:
            mask = (u / smaja[k])**2 + (v / smina[k])**2 <= 1
        else:
            if thin:
                # This is wrong.
                mask = ((u / smaja[k] + 0.4)**2 + (v / smina[k]+0.4)**2 <= 1) & ((u / smaja[k] + 0.4)**2 + (v / smina[k]+0.4)**2 >= 1)
            else:
                if k == 0:
                    mask = (u / smaja[k])**2 + (v / smina[k])**2 <= 1
                else:
                    mask = ((u / smaja[k])**2 + (v / smina[k])**2 <= 1) & ((u / smaja[k-1])**2 + (v / smina[k-1])**2 > 1)
        masks.append(mask)
    
    header = fits.Header()
    # r_e is not effective radius! it is semi-major axis!
    header['BTYPE'] = f"Mask for HLR = {r_e}. 1 in, 0 out."
    header['OBJECT'] = f'NGC{num}'
    header['SIMPLE'] = ('T', 'conforms to FITS standard')
    header['BITPIX'] = (64, 'array data type')
    header['NAXIS'] = (2, 'number of array dimensions')
    header['NAXIS1'] = masks[-1].shape[1]
    header['NAXIS2'] = masks[-1].shape[0]
    header['EXTEND'] = 'T'
    hdu = fits.PrimaryHDU(data=masks[-1].astype(int), header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(f'{out_dir}/mask_re_ngc{num}.fits', overwrite=True)
    # check mask
    # for i, ma in enumerate(masks):
    #     if i%2 == 0:
    #         plt.imshow(ma, origin='lower')
    # plt.savefig(f'../output/images/masks{num}.pdf')
    return center, radii, masks[:-1], masks[-1], (hlr, r_e, r_e * (1 - eps))

def plot_ellipses(image, center, smaja, smina, pa, out_dir, num, radname=''):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(image, norm='log', origin='lower', cmap='cividis')
    plt.scatter(center[0], center[1], color='b', marker='x')
    for i, sma in enumerate(smaja):
        ap = EllipticalAperture(
            positions=(center[0], center[1]),
            a=smaja[i],
            b=smina[i],
            theta=pa          # photutils aperture uses the same PA convention
        )
        if i == len(smaja) - 1:
            ap.plot(ax=ax, color='k', linewidth=1)
        else:
            ap.plot(ax=ax, color='w', linewidth=1)
    
    plt.savefig(f'{out_dir}/sfh_ellipses{radname}_ngc{num}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sfh(sfh, ages, radii, out_dir, num, thin=False, vmin=0.001, vmax=80, hlr=False):
    """
    radii in arcsec
    """
    log_ages = np.log10(ages.value)
    # midpoints between consecutive ages in log space
    log_edges = np.concatenate([
        [log_ages[0] - (log_ages[1] - log_ages[0]) / 2],
        (log_ages[:-1] + log_ages[1:]) / 2,
        [log_ages[-1] + (log_ages[-1] - log_ages[-2]) / 2]
    ])
    edges = 10**log_edges
    y_edges = np.arange(len(radii) + 1) - 0.5

    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.pcolormesh(edges, y_edges, sfh, norm='log', cmap='inferno', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, location='top', fraction=0.15, pad=0.05);
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_title(r'SFR (M$_\odot$/yr)', fontsize=12)
    if len(radii) < 10:
        ax.set_yticks(y_edges, labels=['0'] + [f"{r}" for r in radii]);
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('Lookback time (Gyr)', fontsize=12)
    ax.set_ylabel(f'$R_{{{num}}}$ (arcsec)', fontsize=12)

    if hlr:
        radname = '_HLR'
    else:
        radname = ''

    if thin:
        outname = f'{out_dir}/radial_sfh{radname}_ngc{num}_thin.pdf'
    else:
        outname = f'{out_dir}/radial_sfh{radname}_ngc{num}_thick.pdf'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close()    

def plot_mass(masses, ages, radii, out_dir, num, thin=False, vmin=1e4, vmax=1e10, hlr=False):
    """
    radii in arcsec
    """
    log_ages = np.log10(ages.value)
    # midpoints between consecutive ages in log space
    log_edges = np.concatenate([
        [log_ages[0] - (log_ages[1] - log_ages[0]) / 2],
        (log_ages[:-1] + log_ages[1:]) / 2,
        [log_ages[-1] + (log_ages[-1] - log_ages[-2]) / 2]
    ])
    edges = 10**log_edges
    y_edges = np.arange(len(radii) + 1) - 0.5

    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.pcolormesh(edges, y_edges, masses, norm='log', cmap='inferno', vmin=vmin, vmax=vmax) # 
    cbar = plt.colorbar(im, ax=ax, location='top', fraction=0.15, pad=0.05);
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_title(r'Mass (M$_\odot$)', fontsize=12)
    ax.set_yticks(y_edges, labels=['0'] + [f"{r}" for r in radii]);
    ax.set_xscale('log')
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlabel('Lookback time (Gyr)', fontsize=12)
    ax.set_ylabel('Radius (arcsec)', fontsize=12)

    if hlr:
        radname = '_HLR'
    else:
        radname = ''

    if thin:
        outname = f'{out_dir}/radial_mass{radname}_ngc{num}_thin.pdf'
    else:
        outname = f'{out_dir}/radial_mass{radname}_ngc{num}_thick.pdf'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close()

def save_re(ages, mass_in, mass_in_err, mass_out, mass_out_err, dt_yr, re, out_dir, num):
    """
    radii in arcsec
    """
    sfh_in = mass_in / dt_yr
    sfh_in_err = mass_in_err / dt_yr
    sfh_out = mass_out_err / dt_yr
    sfh_out_err = mass_out_err / dt_yr
    df = pd.DataFrame()
    df['age (Gyr)'] = ages.value
    df[f'M_{re}_in (M_sun)'] = mass_in
    df[f'DM_{re}_in (M_sun)'] = mass_in_err
    df[f'SFR_{re}_in (M_sun/yr)'] = sfh_in
    df[f'DSFR_{re}_in (M_sun/yr)'] = sfh_in_err
    df[f'M_{re}_out (M_sun)'] = mass_out
    df[f'DM_{re}_out (M_sun)'] = mass_out_err
    df[f'SFR_{re}_out (M_sun/yr)'] = sfh_out
    df[f'DSFR_{re}_out (M_sun/yr)'] = sfh_out_err
    df.to_csv(f'{out_dir}/re_mass_sfh_ngc{num}.csv', index=False)

def save_sfh(ages, sfh, sfh_err, radii, out_dir, num, thin=False, hlr=False):
    """
    radii in arcsec
    """
    if hlr:
        radname = '_HLR'
    else:
        radname = ''
    df = pd.DataFrame()
    df['age (Gyr)'] = ages.value
    for i, r in enumerate(radii):
        df[f'SFR_{r} (M_sun/yr)'] = sfh[i]
        df[f'DSFR_{r} (M_sun/yr)'] = sfh_err[i]
    df.to_csv(f'{out_dir}/radial_sfh{radname}_ngc{num}.csv', index=False)

def save_mass(ages, masses, masses_err, radii, out_dir, num, thin=False, hlr=False):
    """
    radii in arcsec
    """
    if hlr:
        radname = '_HLR'
    else:
        radname = ''
    df = pd.DataFrame()
    df['age (Gyr)'] = ages.value
    for i, r in enumerate(radii):
        df[f'M_{r} (M_sun)'] = masses[i]
        df[f'DM_{r} (M_sun)'] = masses_err[i]
    df.to_csv(f'{out_dir}/radial_mass{radname}_ngc{num}.csv', index=False)

def main():
    args = get_args()
    ssp_path = args.ssp_path
    sfh_path = args.sfh_path
    out_dir = args.out_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    num = ssp_path.lower().split('ngc')[-1][:3]
    z = get_z(num)
    delta = 0.2 # arcsec/pixel
    # radii = np.array([1, 2, 5, 10, 20, 30]) # in arcsec
    radii = np.arange(1, 31, 1) # in arcsec

    ssp_data, ssp_header, sfh_data, sfh_header = read_data(ssp_path, sfh_path)
    # values ot of the image are -1, set to 0
    ssp_data[0][ssp_data[0] == -1] = 0
    ages = get_ages(sfh_header)
    sort = np.argsort(ages)
    ages = ages[sort]
    dt = np.gradient(ages).to(u.yr).value
    nbins = len(ages)
    fractions = sum_fractions(sfh_data, nbins)
    fractions = fractions[sort, :, :]
    # delete narrow bin 14 and recalculate nbins and dt
    ages = np.delete(ages, 14)
    fractions = np.delete(fractions, 14, axis=0)
    nbins = len(ages)
    dt = np.gradient(ages).to(u.yr).value
    # Get ellipses for our set radii
    # get masks for rings
    thin=False
    center, radii, masks, mask_re, re = make_ellipses(ssp_data, out_dir, num, radii=radii / delta, thin=thin)
    # radii is returned in pixels
    radii = radii * delta
    # calculate mass and SFR for each ring
    masses = []
    masses_err = []
    for ma in masks:
        _, _, mass, emass = calc_mass(ssp_data, fractions, nbins, z , ma)
        masses.append(mass)
        masses_err.append(emass)
    sfh, sfh_err = masses/dt, masses_err/dt
    
    plot_sfh(sfh, ages, radii, out_dir, num, thin, vmin=0.001, vmax=80)
    plot_mass(masses, ages, radii, out_dir, num, thin, vmin=1e5, vmax=3e9)
    save_sfh(ages, sfh, sfh_err, radii, out_dir, num, thin)
    save_mass(ages, masses, masses_err, radii, out_dir, num, thin)

    # for effective radius HLR:
    _, _, mass_in, emass_in = calc_mass(ssp_data, fractions, nbins, z, mask_re)
    _, _, mass_out, emass_out = calc_mass(ssp_data, fractions, nbins, z, ~mask_re)
    save_re(ages, mass_in, emass_in, mass_out, emass_out, dt, re[0]*delta, out_dir, num)

    # Get ellipses with radii from HLR
    _, radii, masks, _, _ = make_ellipses(ssp_data, out_dir, num, radii=None, thin=thin)
    # we don't mind about units of radii because it is in HLR
    masses = []
    masses_err = []
    for ma in masks:
        _, _, mass, emass = calc_mass(ssp_data, fractions, nbins, z , ma)
        masses.append(mass)
        masses_err.append(emass)
    sfh, sfh_err = masses/dt, masses_err/dt
    
    plot_sfh(sfh, ages, radii, out_dir, num, thin, vmin=0.001, vmax=80, hlr=True)
    plot_mass(masses, ages, radii, out_dir, num, thin, vmin=1e5, vmax=3e9, hlr=True)
    save_sfh(ages, sfh, sfh_err, radii, out_dir, num, thin, hlr=True)
    save_mass(ages, masses, masses_err, radii, out_dir, num, thin, hlr=True)

if __name__ == "__main__":
    main()