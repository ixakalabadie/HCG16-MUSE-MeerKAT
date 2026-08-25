import os
from astropy.io import fits
import configparser
import json
from astropy import units as u
import numpy as np
import argparse

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select dataset'
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('-n', '--number', dest='num', \
    #                     help='The number from the galaxy name (e.g., 838 for NGC838).')
    parser.add_argument('-d', '--datacube', dest='cube_path', \
                        help='Datacube to process.')
    parser.add_argument('-e', '--elines', dest='elines_path', \
                        help='flux_elines output file from PyPipe3D.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    parser.add_argument('-s', '--signal_to_noise', dest='s2n', \
                        help='S/N threshold to create masks.')
    args = parser.parse_args()
    return args

def get_z(num):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get('ngc'+num, "z"))
    return z

def subcube_Ha(in_path, out_path):
    in_name = os.path.basename(in_path)
    num = in_name.lower().split('ngc')[-1][:3]
    os.makedirs(out_path, exist_ok=True)
    out_path = f"{out_path}/{in_name[:-5]}_Halpha.fits"
    with fits.open(in_path) as hdul:
        data = hdul[0].data
        head = hdul[0].header

    z = get_z(num)
    halpha =  6564.614 * u.AA # Halpha rest wavelength in Angstroms
    halpha_obs =  halpha * (1 + z) # observed wavelength in Angstroms
    pix = np.round((halpha_obs.value - head['CRVAL3']) / head['CDELT3']).astype(int)

    subcube = data[pix-30:pix+35]
    head['NAXIS3'] = subcube.shape[0]
    head['CRPIX3'] = head['CRPIX3'] - (pix - 30)

    hdu = fits.PrimaryHDU(subcube, header=head)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(out_path, overwrite=True)

    return out_path

def wave_to_vel(in_path, out_path):
    in_name = os.path.basename(in_path)
    num = in_name.lower().split('ngc')[-1][:3]
    os.makedirs(out_path, exist_ok=True)
    out_path = f"{out_path}/{in_name[:-5]}_vopt.fits"
    with fits.open(in_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    wave_axis = header['CRVAL3'] + header['CD3_3'] * np.arange(1-header['CRPIX3'], header['NAXIS3']-header['CRPIX3']+1)

    halpha =  6564.614 * u.AA # Halpha rest wavelength in Angstroms
    z = get_z(num)
    z = z * u.dimensionless_unscaled
    wave_to_vel = u.doppler_optical(halpha)

    vopt_axis = (wave_axis * u.AA).to(u.m / u.s, equivalencies=wave_to_vel)

    header['CRPIX3'] = 1
    header['CD3_3'] = (vopt_axis[1] - vopt_axis[0]).value
    header['CDELT3'] = (vopt_axis[1] - vopt_axis[0]).value
    header['CRVAL3'] = vopt_axis[0].value
    header['CTYPE3'] = 'VOPT'
    header['CUNIT3'] = 'm/s'
    header['RESTFREQ'] = halpha.to(u.Hz, equivalencies=u.spectral()).value
    header['RESTWAV'] = halpha.value

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(out_path, overwrite=True)

    return out_path

def extract_Ha(in_path, out_path):
    in_name = os.path.basename(in_path)
    num = in_name.lower().split('ngc')[-1][:3]
    os.makedirs(out_path, exist_ok=True)
    out_dir = out_path
    # order is [flux, vopt, disp, ew]
    data_index = (20, 50, 80, 110)
    error_index = (140, 170, 200, 230)
    # Flux in erg/s/cm^2/Angstrom (see header for factor)
    # Velocity in km/s
    # Dispersion in km/s
    # Ha equivalent width in Angstrom
    units = ['10**(-16)*erg/s/cm**2/Angstrom', 'km/s', 'km/s', 'Angstrom']
    mags = ['flux', 'vopt', 'disp', 'EW']

    with fits.open(in_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    paths_out = []

    for i in range(len(units)):
        newheader = fits.Header()
        newheader['SIMPLE'] = header['SIMPLE']
        newheader['BITPIX'] = header['BITPIX']
        newheader['NAXIS'] = (2, 'number of array dimensions')
        newheader['NAXIS1'] = header['NAXIS1']
        newheader['NAXIS2'] = header['NAXIS2']
        newheader['BUNIT'] = units[i]
        newheader['BTYPE'] = mags[i] + ' Ha'
        newheader['EXTNAME'] = 'IMAGE'
        newheader['OBJECT'] = f'NGC{num} (Ha)'
        newheader['CD1_1'] = header['CD1_1']
        newheader['CD1_2'] = header['CD1_2']
        newheader['CD2_1'] = header['CD2_1']
        newheader['CD2_2'] = header['CD2_2']
        newheader['CRPIX1'] = header['CRPIX1']
        newheader['CRPIX2'] = header['CRPIX2']
        newheader['CRVAL1'] = header['CRVAL1']
        newheader['CRVAL2'] = header['CRVAL2']
        newheader['CUNIT1'] = header['CUNIT1']
        newheader['CUNIT2'] = header['CUNIT2']
        newheader['CTYPE1'] = header['CTYPE1']
        newheader['CTYPE2'] = header['CTYPE2']
        # add reference to error in image header
        newheader['EXT_ERR'] = 'ERROR'

        err_header = newheader.copy()
        err_header['BTYPE'] = f'error {mags[i]} Ha'
        err_header['EXTNAME'] = 'ERROR'

        primary_hdu = fits.PrimaryHDU(data[data_index[i]], header=newheader)
        image_hdu = fits.ImageHDU(data[error_index[i]], header=err_header)
        hdulist = fits.HDUList([primary_hdu, image_hdu])
        out_path = f"{out_dir}/NGC{num}_{mags[i]}_Ha.fits"
        hdulist.writeto(out_path, overwrite=True)
        paths_out.append(out_path)

    return paths_out

def make_mask(in_path, out_path, s2n=3, std=0):
    """
    Returns paths of the masked input file and of the mask by itself.

    Parameters
    ----------
    s2n : float
        Signal to noise threshold for the mask. Default is 3.
    std : float
        Standard deviation to apply a Gaussian kernel smoothing to the mask. Default is 0, no smoothing.

    """
    in_name = os.path.basename(in_path)
    num = in_name.lower().split('ngc')[-1][:3]
    os.makedirs(out_path, exist_ok=True)
    out_dir = out_path

    with fits.open(in_path) as hdul:
        data = hdul[0].data
        error = hdul[1].data
        header = hdul[0].header
        err_header = hdul[1].header

    if 'EW' in in_path:
        data[data == 0] = np.nan
    else:
        data[data <= 0] = np.nan

    mask = data/error > s2n

    if std > 0:
        gauss_kernel = Gaussian2DKernel(std)
        mask = convolve(mask, gauss_kernel)

    print(data.shape, np.nansum(mask))

    header['BTYPE'] = (header['BTYPE'], f'With mask of {s2n} S/N and std={std} smoothing')
    err_header['BTYPE'] = (err_header['BTYPE'], f' With mask of {s2n} S/N and std={std} smoothing')

    masked_data = data * mask
    masked_data[masked_data == 0] = np.nan
    masked_err = error * mask
    masked_err[masked_err == 0] = np.nan

    # Write masked image
    hdu = fits.PrimaryHDU(masked_data, header=header)
    err_hdu = fits.ImageHDU(masked_err, header=err_header)
    hdul = fits.HDUList([hdu, err_hdu])
    out_path_masked = f"{out_dir}/{in_name[:-5]}_masked.fits"
    hdul.writeto(out_path_masked, overwrite=True)

    # Write only mask
    header['BITPIX'] = 32
    header['BUNIT'] = ''
    header['BTYPE'] = ('MASK', f'Mask of {s2n} S/N and std={std} smoothing')
    hdu = fits.PrimaryHDU((mask).astype(int), header=header)
    hdul = fits.HDUList([hdu])
    out_path_mask = f"{out_dir}/{in_name[:-5]}_mask.fits"
    hdul.writeto(out_path_mask, overwrite=True)

    return out_path_masked, out_path_mask

def main():
    args = get_args()
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    cube_path = args.cube_path
    elines_path = args.elines_path
    out_dir = args.out_dir
    out_dir = f"{out_dir}"
    s2n = float(args.s2n)

    try:
        subcube_path = subcube_Ha(cube_path, out_dir)
        subcube_vopt = wave_to_vel(subcube_path, out_dir)
    except FileNotFoundError as e:
        print(f"Error processing datacube: {e}")
        pass

    ha_paths = extract_Ha(elines_path, out_dir)
    for pa in ha_paths:
        make_mask(pa, out_dir, s2n=s2n)

if __name__ == "__main__":
    main()
