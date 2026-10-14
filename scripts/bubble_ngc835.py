import os
import argparse
import configparser
import json
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from astropy.convolution import convolve, Gaussian2DKernel

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", "--cube", dest="cube_path", required=True, help="H-alpha vopt datacube")
    parser.add_argument('-e', '--elines', dest='elines_path', \
                        help='flux_elines output file from PyPipe3D.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    args = parser.parse_args()
    return args

def get_z(num):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get('ngc'+num, "z"))
    return z

def load_cube(path):
    with fits.open('../output/ngc835/GAS.NGC835.cube_Halpha_vopt.fits') as hdul:
        data = hdul[0].data
        cube_header = hdul[0].header
        wcs = WCS(cube_header).celestial
    ra = cube_header['CRVAL1'] + cube_header['CD1_1'] * np.arange(1-cube_header['CRPIX1'], cube_header['NAXIS1']-cube_header['CRPIX1']+ 1) # deg
    dec = cube_header['CRVAL2'] + cube_header['CD2_2'] * np.arange(1-cube_header['CRPIX2'], cube_header['NAXIS2']-cube_header['CRPIX2']+ 1) # deg
    vel = cube_header['CRVAL3'] + cube_header['CD3_3'] * np.arange(1-cube_header['CRPIX3'], cube_header['NAXIS3']-cube_header['CRPIX3']+ 1) # m/s
    return data, cube_header, wcs, ra, dec, vel

def plot_bubble_channels(data, wcs, v1, v2, cx, cy, r, out_dir, cmap='inferno'):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(cmap(0.0))

    fig, ax = plt.subplots(ncols=3, figsize=(16,7), subplot_kw=dict(projection=wcs))
    im = ax[0].imshow(data[v2], origin='lower', norm='log', cmap=cmap, vmin=0.0001)
    ax[0].set_xlim(200,300)
    ax[0].set_ylim(220, 300)
    ax[0].scatter(cx, cy)
    circ = Circle((cx, cy), r, facecolor='none', edgecolor='w')
    ax[0].add_patch(circ);
    im = ax[1].imshow(data[int(v1+(v2-v1)/2+1)], origin='lower', norm='log', cmap=cmap, vmin=0.0001)
    ax[1].set_xlim(200,300)
    ax[1].set_ylim(220, 300)
    ax[1].scatter(cx, cy)
    circ = Circle((cx, cy), r, facecolor='none', edgecolor='w')
    ax[1].add_patch(circ);
    ax[2].imshow(data[v1], origin='lower', norm='log', cmap=cmap, vmin=0.0001)
    ax[2].set_xlim(200,300)
    ax[2].set_ylim(220, 300)
    ax[2].scatter(cx, cy)
    circ = Circle((cx, cy), r, facecolor='none', edgecolor='w')
    ax[2].add_patch(circ);

    plt.savefig(os.path.join(out_dir, f"bubble_NGC835_channels.pdf"), dpi=300, bbox_inches='tight')

def calc_radius(subcube, v1, v2, sigma=3):
    e1, e2 = [], []
    for i in range(10):
        y = np.nansum(subcube[v1:v2+1,:,250-205+i], axis=0)
        fil = gaussian_filter(y, sigma=sigma)
        ar = argrelextrema(fil, np.greater)[0]
        e1.append(ar[-2])
        e2.append(ar[-1])
    #     plt.plot(fil, label=str(i))
    # plt.legend()
    print("Mean peak positions: ",np.mean(e1), np.mean(e2))
    print("corresponding std: ", np.std(e1), np.std(e2))
    e1_err, e2_err = np.std(e1), np.std(e2)
    e1, e2 = np.mean(e1), np.mean(e2)
    radius = (e2-e1) * u.pix / 2
    radius_err = np.sqrt(e2_err**2 + e1_err**2) * u.pix / 2
    return radius, radius_err, e1, e2

def plot_dec_mean_spectrum(subcube, e1, e2, out_dir, sigma=3):
    fig, ax = plt.subplots(figsize=(12,4))
    y = np.nansum(subcube[22:30,:,250-200], axis=0)
    fil = gaussian_filter(y, sigma=3)
    ax.plot(y, label="Sum spectrum")
    ax.plot(fil, label="Smoothed")
    ax.axvline(e1, c='k')
    ax.axvline(e2, c='k')
    ax.set_xlabel('Distance in Dec. (pix)')
    plt.legend()

    plt.savefig(os.path.join(out_dir, f"bubble_NGC835_spectrum.pdf"), dpi=300, bbox_inches='tight')

def calc_speed(vel, v1, v2):
    speed = (vel[v2] - vel[v1]) * u.Unit('m/s') / 2 # bubble expands both ways
    # err_speed = np.sqrt(2*(vel[v2] - vel[v2-1])**2) / 2 * u.Unit('m/s') # 1/2*sqrt(d1**2+d2**2)
    speed_err = (vel[v2] - vel[v2-1]) * u.Unit('m/s') # more conservative
    return speed, speed_err


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

def load_elines(path):
    with fits.open(path) as hdul:
        head = hdul[0].header
        wcs = WCS(head).celestial
        sii_16, e_sii_16 = hdul[0].data[24] , hdul[0].data[144] # [SII]6717
        sii_31, e_sii_31 = hdul[0].data[25], hdul[0].data[145] # [SII]6731
        # nii = hdul[0].data[21] # [NII]6584
        # ha = hdul[0].data[20] # H-alpha

    # Process
    sii_16[sii_16 <= 0] = np.nan
    sii_31[sii_31 <= 0] = np.nan

    return wcs, sii_16, sii_31

def plot_ha(image, wcs, cx, cy, r, out_dir, cmap='cividis'):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(cmap(0.0))
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    im = ax.imshow(image, norm='log', cmap=cmap, vmin=0.0001, vmax=0.01)
    circ = Circle((cx,cy), r, facecolor='none', edgecolor='w', linewidth=3)
    ax.add_patch(circ);
    ax.set_xlim(200,300)
    ax.set_ylim(220, 300)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flux ($10^{{-16}}$ erg/s/cm$^2$)", rotation=270, fontsize=14, labelpad=20)
    plt.scatter(cx, cy, marker='+', color='w', s=120, linewidths=2);
    ax.set_xlabel('RA', fontsize=16)
    ax.set_ylabel('Dec', fontsize=16)
    ax.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(os.path.join(out_dir, f"bubble_halpha.pdf"), dpi=300, bbox_inches='tight')

def plot_sii_ratio(sii_ratio, wcs, cx, cy, r, gauss, out_dir, cmap='jet'):
    fig, ax = plt.subplots(subplot_kw=dict(projection=wcs))
    im = ax.imshow(convolve(sii_ratio, gauss, preserve_nan=True), norm='linear', cmap=cmap, vmin=0.5, vmax=1.5, origin='lower') # convolve(sii_ratio, gauss, preserve_nan=True)
    circ = Circle((cx, cy), r, facecolor='none', edgecolor='k', linewidth=3)
    ax.add_patch(circ);
    ax.set_xlim(200,300)
    ax.set_ylim(220, 300)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$[\mathrm{SII}]\,\lambda3716/\lambda3731$', rotation=270, fontsize=14, labelpad=20)
    plt.scatter(cx, cy, marker='+', color='k', s=120, linewidths=2);
    ax.set_xlabel('RA', fontsize=16)
    ax.set_ylabel('Dec', fontsize=16)
    ax.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(os.path.join(out_dir, f"bubble_sulphur.pdf"), dpi=300, bbox_inches='tight')

def main():
    args = get_args()
    cube_path = args.cube_path
    elines_path = args.elines_path
    out_dir = args.out_dir
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    cube_data, cube_header, cube_wcs, ra, dec, vel = load_cube(cube_path)

    # The following values have been obtained manually
    subcube = cube_data[:,220:300,200:300] # subcube containing the bubble
    v1, v2 = 23, 28 # start and end of bubble emission
    cx, cy = 250, 271 #estimated centre of the bubble
    r = 12 # estimated radius

    plot_bubble_channels(cube_data, cube_wcs, v1, v2, cx, cy, r, out_dir)

    radius, radius_err, e1, e2 = calc_radius(subcube, v1, v2)
    speed, speed_err = calc_speed(vel, v1, v2)

    plot_dec_mean_spectrum(subcube, e1, e2, out_dir)

    z = get_z('835')
    d = cosmo.luminosity_distance(z).to(u.Mpc).value
    arc_pix = 0.2 * u.arcsec/u.pix
    kpc_arc = cosmo.kpc_proper_per_arcmin(z).to('kpc/arcsec')
    
    print('V = ', speed.to('km/s'), speed_err.to('km/s'))
    radius_kpc = radius*arc_pix*kpc_arc
    radius_kpc_err = radius_err*arc_pix*kpc_arc
    print('R = ', radius_kpc.to('pc'), radius_kpc_err.to('pc'))

    age = radius_kpc / speed
    age_err = age * np.sqrt((radius_kpc_err/radius_kpc)**2 + (speed_err/speed)**2)
    print('Age = ', age.to('Myr'), age_err.to('Myr'))

    text = f"Parameters of the bubble in NGC835\nRadius: {radius_kpc.to('pc'):.5e} \\pm {radius_kpc_err.to('pc'):.5e}\nVelocity: {speed.to('km/s'):.5e} \\pm {speed_err.to('km/s'):.5e}\nAge: {age.to('Myr'):.5f} \\pm {age_err.to('Myr'):.5f}\n\n"

    savefile(text, out_dir, 835, "Parameters of the bubble in", "Age:")

    elines_wcs, sii_16, sii_31 = load_elines(elines_path)

    sii_ratio = (sii_16/sii_31)
    sigma = 2.5
    gauss = Gaussian2DKernel(sigma)

    plot_ha(cube_data[v1], elines_wcs, cx, cy, r, out_dir)

    plot_sii_ratio(sii_ratio, elines_wcs, cx, cy, r, gauss, out_dir)

if __name__ == "__main__":
    main()