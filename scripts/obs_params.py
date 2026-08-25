import os
import argparse
import configparser
import json
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as cosmo


NHI_CONST = 1.104e21
BEAM_FAC = np.pi/ (4 * np.log(2))

def coldens_sensitivity(noise_mjy, linewidth_kms, v_res, bmaj_arcsec, bmin_arcsec,
                        n_sigma=3.0):
    """
    Column density sensitivity over a given line width.

    Parameters
    ----------
    noise_mjy : float
        RMS noise per channel of the cube, in mJy/beam.
    linewidth_kms : float
        Assumed HI line width, in km/s.
    v_res : float
        Velocity resolution, in km/s
    bmaj_arcsec, bmin_arcsec : float
        Beam major and minor axis FWHM, in arcsec.
    n_sigma : float, optional
        Significance level of the limit (default 3).

    Returns
    -------
    float
        N_HI sensitivity in cm^-2.
    """
    # Integrated flux of an n_sigma signal spread over the line width
    # [mJy/beam km/s]
    integrated_flux = n_sigma * noise_mjy * np.sqrt(linewidth_kms * v_res)
    return NHI_CONST * integrated_flux / (bmaj_arcsec * bmin_arcsec)

def hi_mass(mom0, dist, bmaj, bmin, pix1, pix2):
    """
    Total HI mass in the image.

    Parameters
    ----------
    mom0 : 2D array
        Moment 0 map, in Jy/beam·km/s.
    dist : float
        Distance to object, in Mpc.
    bmaj, bmin : float
        Beam major and minor axis FWHM, in arcsec.
    pix1, pix2 : float
        Width and height of a pixel, in arcsec.
    """
    return 2.35e5 * dist**2 * np.nansum(mom0) / (BEAM_FAC * bmaj * bmin / (pix1 * pix2))

def get_data():
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get("general", "z"))
        mom0_path = json.loads(config_object.get("general", "hi_mom0"))
        dz = cosmo.luminosity_distance(z).to('Mpc').value
    
    with fits.open(mom0_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
    return data, header, wcs, z

def resolution(header):
    if 'CUNIT1' not in header.keys():
        skyunit = 'deg'
    else:
        skyunit = header['CUNIT1']
    bmaj = header['BMAJ'] * u.Unit(skyunit).to('arcsec')
    bmin = header['BMIN'] * u.Unit(skyunit).to('arcsec')
    pix1 = np.abs(header['CDELT1']) * u.Unit(skyunit).to('arcsec')
    pix2 = np.abs(header['CDELT2']) * u.Unit(skyunit).to('arcsec')
    v_res = 5.7 * u.km/u.s # works both for line60 and highres
    # v_res = np.abs(header['CDELT3']) * u.Unit(header['CUNIT3']).to('km/s')
    return bmaj, bmin, pix1, pix2, v_res

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--catalog', dest='cat_path', \
                        help='Path to SoFiA output catalog.txt.')
    parser.add_argument('-m', '--muse', dest='muse_path', \
                        help='Path to MUSE cube.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders may be created there.')
    args = parser.parse_args()
    return args

def get_noise(cat_path, score_col=12, value_col=23):
    """
    Get rms of the subcube with most pixels from SoFiA catalog. The column of the number of pixels is given by score_col (usually 12) and the rms by value_col (usually 23). Returns noise in Jy/beam
    """
    data = np.genfromtxt(cat_path,
            dtype=str,
            comments="#",
            delimiter=None,
            autostrip=True,
            ndmin=2
        )

    if data.size == 0:
        raise ValueError(f"Empty catalog file: {catalog_path}")
    
    ncols = data.shape[1]
    if score_col >= ncols or value_col >= ncols:
        raise IndexError(
            f"Catalog has {ncols} columns, need cols {score_col} and {value_col}"
        )
    
    scores = []
    for x in data[:, score_col]:
        try:
            scores.append(float(x))
        except ValueError:
            scores.append(np.nan)
    scores = np.array(scores)

    if np.all(np.isnan(scores)):
        raise ValueError(f"No valid scores in catalog: {catalog_path}")

    best_row = int(np.nanargmax(scores))
    return float(data[best_row, value_col])

def savefile(text, out_dir, sstr, estr):
    if not os.path.exists(f'{out_dir}/obs_params.txt'):
        with open(f'{out_dir}/obs_params.txt', 'w') as f:
            f.write(text)
        return
    with open(f'{out_dir}/obs_params.txt', 'r') as f:
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
    
    with open(f'{out_dir}/obs_params.txt', 'w') as f:
        f.writelines(lines)

def main():
    args = get_args()
    cat_path = args.cat_path
    muse_path = args.muse_path
    out_dir = args.out_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    mom0_data, mom0_header, wcs, z = get_data()
    if mom0_header['BUNIT'] == 'Jy/beam*m/s':
        mom0_header['BUNIT'] = 'Jy*m/beam/s'
    mom0_data = mom0_data * u.Unit(mom0_header['BUNIT']).to(u.Jy/u.beam*u.km/u.s)
    dist = cosmo.luminosity_distance(z).to('Mpc')

    bmaj, bmin, pix1, pix2, v_res = resolution(mom0_header)

    noise = get_noise(cat_path) * 1000

    himass = hi_mass(mom0_data, dist.value, bmaj, bmin, pix1, pix2)

    nhi_sens = coldens_sensitivity(noise, 20, v_res, bmaj, bmin, n_sigma=3).value

    linres = np.sqrt(mom0_header['BMAJ']*mom0_header['BMIN']) * u.deg
    linres = (linres * dist).to(u.kpc, equivalencies=u.dimensionless_angles())

    halpha =  6564.614 * u.AA
    wave_to_vel = u.doppler_optical(halpha)

    text = f"Observational parameters\nDistance: {dist.value:.5f} Mpc\nLinear Resolution: {linres.value:.5f} kpc\nNoise: {noise:.5f} mJy/beam\nColum density 3sigma sensitivity: {nhi_sens:.3e} cm-2\nHI mass: {himass/1e9:.3f} 1e9 M_sun\n\n"
    print(text)

    savefile(text, out_dir, "Observational param", "HI mass:")


if __name__ == "__main__":
    main()