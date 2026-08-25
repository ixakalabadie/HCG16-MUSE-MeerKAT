import os
import argparse
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import add_beam, add_scalebar
from astropy.cosmology import Planck18 as cosmo
from astropy.convolution import convolve, Gaussian2DKernel
import configparser
import json
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select output directory.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', dest='in_path', \
                        help='Path to Ha velocity data.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    args = parser.parse_args()
    return args

def scalebar_angle(d, bar_len = 10 * u.kpc):
    return (bar_len / d).to(u.deg, equivalencies=u.dimensionless_angles())

def get_center(out_dir, num):
    with open(f"{out_dir}/SFR_ngc{num}.txt", "r") as f:
        for line in f:
            if line.startswith("Center (x,y):"):
                coords = line.split(":")[1].strip()
                x, y = [int(v.strip()) for v in coords.split(",")] 
                break
    return x, y

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
        facecolor='white',
        edgecolor='white',
        zorder=5
    )
    ax.add_patch(ellipse)

def plot_ha_mom1_hi_mom1(in_path, num, out_dir):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get(f"ngc{num}", "z"))
        levels = json.loads(config_object.get(f"ngc{num}", "levels"))
        levels = np.arange(levels[0], levels[1], levels[2])
        hi_mom1_path = json.loads(config_object.get('general', "hi_mom1"))
        halevels = json.loads(config_object.get(f"ngc{num}", "halevels"))
        seeing = json.loads(config_object.get(f"ngc{num}", "seeing"))

    d = cosmo.luminosity_distance(z).to(u.Mpc).value
    try:
        cent = get_center(out_dir, num)
    except Exception as e:
        cent = 0,0

    with fits.open(hi_mom1_path) as hdul:
        mom1_data = hdul[0].data
        mom1_header = hdul[0].header
        mom1_wcs = WCS(mom1_header)

    mom1_data = mom1_data * u.Unit(mom1_header['BUNIT']).to('km/s')

    sigma = 0.8    
    gauss = Gaussian2DKernel(sigma)
    con_data = convolve(mom1_data, gauss, preserve_nan=True)

    with fits.open(in_path) as hdul:
        vel_data = hdul[0].data
        vel_header = hdul[0].header
        vel_wcs = WCS(vel_header)

    vel_data = vel_data * u.Unit(vel_header['BUNIT']).to('km/s')
    vel_data[vel_data <= 0] = np.nan

    cmap = "coolwarm"

    z = z * u.dimensionless_unscaled
    v = cosmo.comoving_distance(z) * cosmo.H(0)
    # if num == '833':
    #     v = 3850 * u.km/u.s
    # if num == '835':
    #     v = 4075 * u.km/u.s
    
    norm = mcolors.TwoSlopeNorm(vcenter=v.value, vmin=halevels[0], vmax=halevels[1])

    fig, ax = plt.subplots(subplot_kw=dict(projection=vel_wcs), figsize=(8,8))
    # ax.set_title(f"hi levels = {levels}")
    ha = ax.imshow(vel_data, cmap=cmap, norm=norm)
    ax.scatter(cent[0], cent[1], color='k', marker='+', s=100, linewidths=2)
    cbar = plt.colorbar(ha, ax=ax, shrink=0.83, pad=0.02);
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(f"Velocity (km/s)", rotation=270, fontsize=14, labelpad=20)
    m0 = ax.contour(con_data, levels=levels, colors='k', transform=ax.get_transform(mom1_wcs)) # colors='deepskyblue'
    plt.xlim(0,vel_data.shape[1])
    plt.ylim(0,vel_data.shape[0])
    add_beam(ax, major=mom1_header['BMAJ'] * u.deg, minor=mom1_header['BMIN'] * u.deg, angle=mom1_header['BPA'], color='k')
    second_beam(ax, vel_wcs, seeing * u.arcsec, seeing * u.arcsec, 0)
    add_scalebar(ax, scalebar_angle(d * u.Mpc, 5*u.kpc), label="5 kpc", color="k", label_top=True, corner="top left")
    
    ax.set_xlabel("RA", fontsize=16)
    ax.set_ylabel("Dec", fontsize=16)
    ax.tick_params(labelsize=14)

    plt.savefig(f"{out_dir}/ha_mom1_hi_mom1_ngc{num}.pdf", dpi=300, bbox_inches='tight')

def main():
    args = get_args()
    in_path = args.in_path
    out_dir = args.out_dir
    num = out_dir.lower().split('ngc')[-1][:3]
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    plot_ha_mom1_hi_mom1(in_path, num, out_dir)

if __name__ == "__main__":
    main()