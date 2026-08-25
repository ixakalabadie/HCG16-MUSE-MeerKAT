import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from lmfit.models import GaussianModel
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
import configparser
import json
from scipy.stats import norm
import argparse

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select output directory.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--datacube', dest='cube_path', \
                        help='Datacube to process.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    args = parser.parse_args()
    return args

def fit_ngaus(x, y, ncomp, guess, noise=None):

    model = None
    params = None

    for i in range(ncomp):
        prefix = f'g{i}_'
        g = GaussianModel(prefix=prefix)

        if model is None:
            model = g
            params = g.make_params()
        else:
            model += g
            params.update(g.make_params())

        # --- Initial guesses ---
        params[f'{prefix}center'].set(value=guess[0], min=guess[1], max=guess[2])
        params[f'{prefix}sigma'].set(value=2*57, min=10, max=100)
        params[f'{prefix}amplitude'].set(value=1, min=0.25) # np.max(y)/ncomp

    result = model.fit(y, params, x=x, weights=None if noise is None else 1/noise)

    return result

def get_z(num):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get('ngc'+num, "z"))
    return z

def fit_spectrum(coords):
    j, i = coords
    spectrum = subcube[:,j,i]

    if np.isnan(spectrum).any():
        return None
    if np.nanmax(spectrum) < 5 * rms:
        return None

    amps = np.zeros(3)
    centers = np.zeros(3)
    sigmas = np.zeros(3)
    bics = np.zeros(3)
    chis = np.zeros(3)

    res1 = fit_ngaus(vel, spectrum, 1, (halpha_v_obs, halpha_v_obs-350, halpha_v_obs+350))
    res2 = fit_ngaus(vel, spectrum, 2, (halpha_v_obs, halpha_v_obs-350, halpha_v_obs+350))
    bics[0] = res1.bic
    bics[1] = res2.bic
    chis[0] = res1.redchi
    chis[1] = res2.redchi

    if res1.bic - res2.bic < 5 and res1.redchi < res2.redchi/0.8:
        res = res1
        n = 1
        if is_significant(res, n, rms, 3) == False:
            return None
    else:
        res3 = fit_ngaus(vel, spectrum, 3, (halpha_v_obs,halpha_v_obs-350,halpha_v_obs+350))
        bics[2] = res3.bic
        chis[2] = res3.redchi
        if res2.bic - res3.bic < 5 and res2.redchi < res3.redchi/0.8:
            res = res2
            n = 2
            if is_significant(res, n, rms, 3) == False:
                return None
        else:
            res = res3
            n = 3
            if is_significant(res, n, rms, 3) == False:
                return None

    for k in range(n):
            amps[k] = res.params[f'g{k}_amplitude'].value
            centers[k] = res.params[f'g{k}_center'].value
            sigmas[k] = res.params[f'g{k}_sigma'].value

    return(j, i, n, amps, centers, sigmas, bics, chis)

def get_halpha_v_obs(num):
    halpha =  6564.614 * u.AA
    z = get_z(num)
    z = z * u.dimensionless_unscaled
    halpha_obs = halpha * (1+z)
    wave_to_vel = u.doppler_optical(halpha)
    halpha_v_obs = (halpha_obs).to(u.km / u.s, equivalencies=wave_to_vel).value
    return halpha_v_obs

def is_significant(res, n, rms, s2n=3):
    """Check if all Gaussian components are above 3*rms peak."""
    for k in range(n):
        amp = res.params[f'g{k}_amplitude'].value
        sig = res.params[f'g{k}_sigma'].value
        peak = amp / (sig * np.sqrt(2*np.pi))
        if peak < s2n * rms:
            return False
    return True

def init_worker(_subcube, _vel, _rms, _halpha_v_obs):
    global subcube, vel, rms, halpha_v_obs
    subcube = _subcube
    vel = _vel
    rms = _rms
    halpha_v_obs = _halpha_v_obs


def main():
    args = get_args()
    cube_path = args.cube_path
    out_dir = args.out_dir
    num = out_dir.lower().split('ngc')[-1][:3]
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    halpha_v_obs = get_halpha_v_obs(num)

    with fits.open(cube_path) as hdul:
        cube_data = hdul[0].data
        cube_header = hdul[0].header
        cube_wcs = WCS(cube_header).celestial

    vel = cube_header['CRVAL3'] + cube_header['CDELT3'] * np.arange(1-cube_header['CRPIX3'], cube_header['NAXIS3']-cube_header['CRPIX3']+ 1)
    vel = vel * u.Unit(cube_header['CUNIT3']).to('km/s')

    _, rms = norm.fit(np.hstack([cube_data[cube_data < 0].flatten(), -cube_data[cube_data < 0].flatten()]))

    subcube = cube_data # [:,150:170,220:240]
    nz, ny, nx = subcube.shape

    amplitude = np.zeros((3, ny, nx))
    center    = np.zeros((3, ny, nx))
    sigma     = np.zeros((3, ny, nx))
    ncomp     = np.zeros((ny, nx), dtype=int)
    bic       = np.zeros((3, ny, nx))
    chi       = np.zeros((3, ny, nx))

    coords = [(j, i) for j in range(ny) for i in range(nx)]

    with Pool(cpu_count(),
            initializer=init_worker,
            initargs=(subcube, vel, rms, halpha_v_obs)
            ) as pool:
        results = list(tqdm(pool.imap_unordered(fit_spectrum, coords),
                        total=len(coords)))

    for result in results:
        if result is None:
            continue

        j, i, n, amps, centers, sigmas, bics, chis = result

        ncomp[j,i] = n
        amplitude[:,j,i] = amps
        center[:,j,i] = centers
        sigma[:,j,i] = sigmas
        bic[:,j,i] = bics
        chi[:,j,i] = chis
    
    print(np.max(center), np.max(sigma))

    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(amplitude, header=cube_header, name="AMPLITUDE"),
        fits.ImageHDU(center, header=cube_header, name="CENTER"),
        fits.ImageHDU(sigma, header=cube_header, name="SIGMA"),
        fits.ImageHDU(ncomp, header=cube_header, name="NCOMP"),
        fits.ImageHDU(bic, header=cube_header, name="BIC"),
        fits.ImageHDU(chi, header=cube_header, name="CHI2")])

    hdul.writeto(f"{out_dir}/gaussian_fits_ngc{num}.fits", overwrite=True)
    print("FITS file saved successfully.")

if __name__ == "__main__":
    main()
