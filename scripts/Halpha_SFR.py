import os
import argparse
import configparser
import json
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as cosmo
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
import matplotlib.ticker as mticker

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-a', '--ha_path', dest='ha_path', \
                        help='Path to Halpha image.')
    parser.add_argument('-b', '--hb_path', dest='hb_path', \
                        help='Path to Hbeta image.')
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

def kennicut_factor():
    return 7.9e-42 * (u.M_sun / u.yr * u.s / u.erg)

def read_data(ha_path, hb_path=None):
    with fits.open(ha_path) as hdul:
        ha_data = hdul[0].data
        ha_err = hdul[1].data
        ha_head = hdul[0].header
        un = 1
        if ha_head['BUNIT'] == ' 10^-16 erg' or ha_head['BUNIT'] == '10**(-16)*erg/s/cm**2/Angstrom':
            un = 1e-16
        ha_data *= un
        ha_err *= un
    with fits.open(hb_path) as hdul:
        hb_data = hdul[0].data[3]
        hb_err = hdul[0].data[123]
        hb_head = hdul[0].header
        un = 1
        if hb_head['UNITS'] == '10^-16 erg' or hb_head['UNITS'] == '1e-16 erg' or hb_head['UNITS'] == '10**(-16)*erg/s/cm**2/Angstrom':
            un = 1e-16
        hb_data *= un
        hb_err *= un
    return ha_data, ha_err, hb_data, hb_err, ha_head

def mask_data(data, error, s2n=3):
    mask = data/error > s2n
    return mask

def sum_data(data, error):
    tot_flux = np.nansum(data)
    tot_err = np.sqrt(np.nansum(error**2))
    return tot_flux, tot_err

def correct_extinction(ha_data, ha_err, hb_data, hb_err):
    ha_wave = 0.656461 # um
    hb_wave = 0.486268 # um
    # from Vogt+2013, appendix A
    ev = -4.61777 + 1.41612 / ha_wave + 1.52077 / ha_wave**2 - 0.63269 / ha_wave**3 + 0.07386 / ha_wave**4
    k = 0.76 * (ev + 4.5)
    r = ha_data / hb_data
    er = r * np.sqrt((ha_err / ha_data)**2 + (hb_err / hb_data)**2)
    # The 2.85 (assuming case B recombination) maybe should be changed towards 3.1 for AGN effects. 4.5 also could be lowered.
    corr_ha = ha_data * (r/2.85)**k
    # error for exact r
    # corr_err = ha_err * (r/2.85)**k
    # error for error in r=ha/hb and ha_data
    corr_err = corr_ha * np.sqrt((1+k)**2 * (ha_err/ha_data)**2 + k**2 * (hb_err/hb_data)**2)
    return corr_ha, corr_err, ev, r, er

def calc_sfr(ha_flux, ha_err, dist):
    lum, elum = flux_to_lum(ha_flux, ha_err, dist)
    sfr, esfr = kennicut_factor() *lum, kennicut_factor() *elum
    return lum, elum, sfr, esfr

def plot_extinction(ha_data, hb_data, header, out_dir, num):
    ha_data[ha_data <= 0] = np.nan
    hb_data[hb_data <= 0] = np.nan
    wcs = WCS(header)
    # from Vogt+2013, appendix A
    av = 8.55 * np.log10(ha_data/hb_data/2.85)

    cmap = "cividis"
    # managua also looks good
    # cmap = plt.get_cmap(cmap).copy()
    # cmap.set_bad(cmap(0.0))
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs), figsize=(8,8))
    im = ax.imshow(av, cmap=cmap, vmin=0)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(r"$A_V$ (mag)", rotation=270, fontsize=14, labelpad=20)
    ax.set_xlabel("RA", fontsize=14)
    ax.set_ylabel("Dec", fontsize=14)
    ax.tick_params(labelsize=12)

    plt.savefig(f"{out_dir}/extinction_ngc{num}.pdf", dpi=300, bbox_inches='tight')

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

def main():
    args = get_args()
    ha_path = args.ha_path
    hb_path = args.hb_path
    out_dir = args.out_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    num = ha_path.lower().split('ngc')[-1][:3]

    ha_data, ha_err, hb_data, hb_err, header = read_data(ha_path, hb_path)
    ha_mask = mask_data(ha_data, ha_err, s2n=3)
    hb_mask = mask_data(hb_data, hb_err, s2n=3)
    mask = ha_mask & hb_mask

    plot_extinction(ha_data*mask, hb_data*mask, header, out_dir, num)

    # ha, eha = sum_data(ha_data, ha_err)
    # hb, ehb = sum_data(hb_data, hb_err)
    ha, eha = sum_data(ha_data[mask], ha_err[mask])
    hb, ehb = sum_data(hb_data[mask], hb_err[mask])

    ha_corr, eha_corr, ev, r, er = correct_extinction(ha, eha, hb, ehb)
    z = get_z(num)
    dist = cosmo.luminosity_distance(z).to(u.Mpc).value
    lum, elum, sfr, esfr = calc_sfr(ha_corr, eha_corr, dist)

    text = f"H-alpha flux and SFR results for NGC{num}\nDistance: {dist:.5f} Mpc\nHalpha flux: {ha:.5e} \\pm {eha:.5e} erg/s/cm^2\nHbeta flux: {hb:.5e} \\pm {ehb:.5e} erg/s/cm^2\nBalmer decrement (R = Ha/Hb): {r:.5f} \\pm {er:.5f}\nHalpha corrected flux: {ha_corr:.5e} \\pm {eha_corr:.5e} erg/s/cm^2\nHalpha luminosity: {lum.value:.5e} \\pm {elum.value:.5e} erg/s\nHalpha SFR: {sfr.value:.5e} \\pm {esfr.value:.5e} M_sun/yr\n\n"

    savefile(text, out_dir, num, "H-alpha flux and SFR", "Halpha SFR:")

if __name__ == "__main__":
    main()