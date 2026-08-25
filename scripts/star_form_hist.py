import os
import argparse
import configparser
import json
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-p', '--ssp', dest='ssp_path', \
                        help='Path to SSP data.')
    parser.add_argument('-f', '--sfh', dest='sfh_path', \
                        help='Path to SFH data.')
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

def plot_sfh(mass, emass, bins, out_dir, num):
    bin_centers, edges, dt, dt_yr = bins
    plt.figure(figsize=(14,5))
    plt.title(f"NGC{num}")
    # Step plot
    plt.step(edges[:-1], mass/dt_yr, where='post', color='blue', linewidth=2, label='SFR')
    # Filled area under the curve
    plt.fill_between(edges[:-1], mass/dt_yr, step='post', alpha=0.3, color='blue')
    plt.errorbar(bin_centers, mass/dt_yr, yerr=emass/dt_yr, fmt='none', color='k', label='Error', capsize=3)
    # Log-scale x-axis
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Lookback time (Myr)', fontsize=12)
    plt.ylabel(r'SFR (M$_\odot$/yr)', fontsize=12);
    plt.tick_params(labelsize=12)

    plt.savefig(f"{out_dir}/SFH_ngc{num}.pdf", dpi=300, bbox_inches='tight')

def save_sfh(mass, emass, ages, dt_yr, out_dir, num):
    df = pd.DataFrame(columns=['age (Gyr)', 'M (M_sun)', 'DM (M_sun)', 'SFR (M_sun/yr)', 'DSFR (M_sun/yr)'])
    df['age (Gyr)'] = ages.value
    df['M (M_sun)'] = mass
    df['DM (M_sun)'] = emass
    df['SFR (M_sun/yr)'] = mass/dt_yr
    df['DSFR (M_sun/yr)'] = emass/dt_yr
    df.to_csv(f"{out_dir}/SFH_ngc{num}.csv", index=False)

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

def ha_equivalent_sfr(ages, mass, emass, out_dir, num):
    idx10 = np.where(ages.to('Myr').value <= 10)[0][-1]
    idx20 = np.where(ages.to('Myr').value <= 20)[0][-1]

    bin10 = (ages[idx10] + (ages[idx10+1] - ages[idx10])/2)
    bin20 = (ages[idx20] + (ages[idx20+1] - ages[idx20])/2)

    sfr_10 = np.sum(mass[:idx10]) / bin10.to('yr').value
    sfr_20 = np.sum(mass[:idx20]) / bin20.to('yr').value

    text = f"Synthesis results for NGC{num}\nSFR_10 ({bin10.to('Myr')}): {sfr_10:.3f} M_sun/yr\nSFR_20 ({bin20.to('Myr')}): {sfr_20:.3f} M_sun/yr\n\n"

    savefile(text, out_dir, num, "Synthesis results", "SFR_20")

def main():
    args = get_args()
    ssp_path = args.ssp_path
    sfh_path = args.sfh_path
    out_dir = args.out_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    num = ssp_path.lower().split('ngc')[-1][:3]
    z = get_z(num)

    ssp_data, ssp_header, sfh_data, sfh_header = read_data(ssp_path, sfh_path)
    ages = get_ages(sfh_header)
    sort = np.argsort(ages)
    ages = ages[sort]
    nbins = len(ages)
    fractions = sum_fractions(sfh_data, nbins)
    fractions = fractions[sort, :, :]
    # don't save luminosity
    _, _, mass, emass = calc_mass(ssp_data, fractions, nbins, z)
    bins = get_bins(ages)

    plot_sfh(mass, emass, bins, out_dir, num)
    save_sfh(mass, emass, ages, bins[3], out_dir, num)
    ha_equivalent_sfr(ages, mass, emass, out_dir, num)

if __name__ == "__main__":
    main()