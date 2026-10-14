import os
import configparser
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from astropy.visualization.wcsaxes import add_beam, add_scalebar
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
import argparse
from matplotlib.patches import Ellipse

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    Inputs are defined in hcg16_gals_params.ini file.
    """
    description = 'Select output directory.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', dest='in_path', \
                        help='Path to input DeCaLS rgz image.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    args = parser.parse_args()
    return args

def scalebar_angle(d, bar_len = 10 * u.kpc):
    return (bar_len / d).to(u.deg, equivalencies=u.dimensionless_angles())

def second_beam(ax, wcs, bmaj, bmin, pa):
    # 2. Get the center of the radio beam in display coordinates
    beam_artist = ax.artists[-1]  # add_beam appends it as the last artist

    # 3. Get the center of the radio beam in display coordinates
    renderer = ax.figure.canvas.get_renderer()
    bbox = beam_artist.get_window_extent(renderer)
    cx = (bbox.x0 + bbox.x1) / 2
    cy = (bbox.y0 + bbox.y1) / 2

    # 4. Convert to axes-fraction coordinates
    ax_bbox = ax.get_window_extent(renderer)
    cx_ax = (cx - ax_bbox.x0) / ax_bbox.width
    cy_ax = (cy - ax_bbox.y0) / ax_bbox.height

    # 5. Convert optical beam size from arcsec to axes-fraction units
    pix_scale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
    img_w = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    img_h = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

    w_ax = (bmin.to(u.arcsec).value / pix_scale) / img_w
    h_ax = (bmaj.to(u.arcsec).value / pix_scale) / img_h

    # 6. Draw optical beam centered on the radio beam
    ellipse = Ellipse(
        xy=(cx_ax, cy_ax),
        width=w_ax,
        height=h_ax,
        angle=pa,
        transform=ax.transAxes,
        facecolor='k',
        edgecolor='k',
        zorder=5
    )
    ax.add_patch(ellipse)

def plot_decals_hi60(out_dir, in_path=None):
    # read paths and parameters from ini file
    config_object = configparser.ConfigParser()
    with open("./hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get(f"general", "z"))
        decals_path = json.loads(config_object.get("general", "decals"))
        mom060_path = json.loads(config_object.get("general", "hi60_mom0"))
        mom0_path = json.loads(config_object.get("general", "hi_mom0"))

    d = cosmo.luminosity_distance(z).to(u.Mpc).value

    with fits.open(mom060_path) as hdul:
        mom060_data = hdul[0].data
        mom060_header = hdul[0].header
        mom060_wcs = WCS(mom060_header)

    with fits.open(mom0_path) as hdul:
        mom0_data = hdul[0].data
        mom0_header = hdul[0].header
        mom0_wcs = WCS(mom0_header)

    if in_path is not None:
        decals_path = in_path
    with fits.open(decals_path) as hdul:
        opt_data = hdul[0].data
        opt_header = hdul[0].header
        opt_wcs = WCS(opt_header).celestial

    # in header grz = [0,1,2]
    # z = 8931 AA, r = 6165 AA, g = 4686 AA
    opt_rgb = make_lupton_rgb(image_r=opt_data[2], image_g=opt_data[1], image_b=opt_data[0], stretch=0.1, Q=10, minimum=0) # [min_r, min_g, min_z]

    mom060_beam_fac = np.pi/ (4 * np.log(2))
    mom060_bmaj = mom060_header['BMAJ'] * 3600  # arcsec
    mom060_bmin = mom060_header['BMIN'] * 3600  # arcsec
    mom060_pix1 = np.abs(mom060_header['CDELT1']) * 3600  # arcsec
    mom060_pix2 = np.abs(mom060_header['CDELT2']) * 3600  # arcsec

    mom0_beam_fac = np.pi/ (4 * np.log(2))
    mom0_bmaj = mom0_header['BMAJ'] * 3600  # arcsec
    mom0_bmin = mom0_header['BMIN'] * 3600  # arcsec
    mom0_pix1 = np.abs(mom0_header['CDELT1']) * 3600  # arcsec
    mom0_pix2 = np.abs(mom0_header['CDELT2']) * 3600  # arcsec

    # Calculate column density
    # Data (flux) in Jy/beam·(m/s) and beam in arcsec
    mom060_nhi = 1.104e+21 * mom060_data/ (mom060_bmaj * mom060_bmin)

    mom0_nhi = 1.104e+21 * mom0_data/ (mom0_bmaj * mom0_bmin)

    fig, ax = plt.subplots(subplot_kw=dict(projection=opt_wcs), figsize=(8,8))
    opt = ax.imshow(opt_rgb)
    # Column density levels from Ianjamasimanana et al. 2025
    #, 1.2e19, 2.5e19, 5e19, 9.9e19, 2e20, 7e20
    m060 = ax.contour(mom060_nhi, levels=[3.1e18, 5e19], colors='purple', transform=ax.get_transform(mom060_wcs))
    m0 = ax.contour(mom0_nhi, levels=[1.5e20, 1e21], colors='dodgerblue', transform=ax.get_transform(mom0_wcs))
    plt.xlim(150,2700)
    plt.ylim(150,2700)
    add_beam(ax, major=mom060_header['BMAJ'] * u.deg, minor=mom060_header['BMIN'] * u.deg, angle=mom060_header['BPA'], color='w')
    second_beam(ax, mom0_wcs, mom0_header['BMAJ'] * u.deg, mom0_header['BMIN'] * u.deg, mom0_header['BPA'])
    add_scalebar(ax, scalebar_angle(d*u.Mpc, 50*u.kpc), label="50 kpc", color="w", label_top=False)
    plt.xlabel('RA', fontsize=18)
    plt.ylabel('Dec', fontsize=18)
    plt.tick_params(labelsize=16)

    ax.text(0.82, 0.68, f"A", c='w', fontsize=20, transform=ax.transAxes)
    ax.text(0.89, 0.80, f"B", c='w', fontsize=20, transform=ax.transAxes)
    ax.text(0.65, 0.75, f"C", c='w', fontsize=20, transform=ax.transAxes)
    ax.text(0.58, 0.63, f"D", c='w', fontsize=20, transform=ax.transAxes)
    ax.text(0.35, 0.2, f"NGC 848", c='w', fontsize=20, transform=ax.transAxes)

    if not os.path.isdir(f"{out_dir}/images"):
        os.makedirs(f"{out_dir}/images")
    plt.savefig(f"{out_dir}/images/hcg16_line60_decals.pdf", dpi=300, bbox_inches='tight')

def main():
    args = get_args()
    in_path = args.in_path
    out_dir = args.out_dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    plot_decals_hi60(out_dir)

if __name__ == "__main__":
    main()
