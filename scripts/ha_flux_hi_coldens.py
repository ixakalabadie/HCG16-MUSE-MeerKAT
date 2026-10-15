import os
import argparse
import configparser
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.visualization.wcsaxes import add_beam, add_scalebar
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
import matplotlib.ticker as mticker
from matplotlib.patches import Ellipse
from matplotlib.font_manager import FontProperties

LAB = {'833': 'B', '835': 'A', '838': 'C', '839': 'D', '848': '848'}

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select output directory.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', dest='in_path', \
                        help='Path to Ha flux data.')
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
        facecolor='k',
        edgecolor='k',
        zorder=5
    )
    ax.add_patch(ellipse)

def plot_ha_flux_hi_coldens(in_path, num, out_dir):
    config_object = configparser.ConfigParser()
    with open("./hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        z = json.loads(config_object.get(f"ngc{num}", "z"))
        mom0_path = json.loads(config_object.get("general", "hi_mom0"))
        nhi_levels = json.loads(config_object.get(f"ngc{num}", "nhi_levels"))
        seeing = json.loads(config_object.get(f"ngc{num}", "seeing"))
    
    d = cosmo.luminosity_distance(z).to(u.Mpc).value

    try:
        cent = get_center(out_dir, num)
    except:
        cent = 0,0

    with fits.open(mom0_path) as hdul:
        mom0_data = hdul[0].data
        mom0_header = hdul[0].header
        mom0_wcs = WCS(mom0_header)

    beam_fac = np.pi/ (4 * np.log(2))
    bmaj = mom0_header['BMAJ'] * 3600  # arcsec
    bmin = mom0_header['BMIN'] * 3600  # arcsec
    pix1 = np.abs(mom0_header['CDELT1']) * 3600  # arcsec
    pix2 = np.abs(mom0_header['CDELT2']) * 3600  # arcsec

    # Data (flux) in Jy/beam·(m/s) and beam in arcsec
    nhi = 1.104e+21 * mom0_data / (bmaj * bmin)

    colormap = "inferno" #Grays??
    cmap = plt.get_cmap(colormap).copy()
    cmap.set_bad(cmap(0.0))

    with fits.open(in_path) as hdul:
        ha_data = hdul[0].data
        ha_header = hdul[0].header
        ha_wcs = WCS(ha_header)

    bunit = ha_header['BUNIT']
    if bunit == '10**(-16)*erg/s/cm**2/Angstrom':
        bunit = r"$10^{-16}\,$erg/s/cm$^2$/Å"

    fig, ax = plt.subplots(subplot_kw=dict(projection=ha_wcs), figsize=(8,8))
    if num != '848':
        ax.set_title(f'HCG 16{LAB[str(num)]}', fontsize=16)
    else:
        ax.set_title(f'NGC {num}', fontsize=16)
    ha = ax.imshow(ha_data, cmap=cmap, norm='log', vmin=0.001, vmax=np.max(ha_data)*0.5)
    ax.scatter(cent[0], cent[1], color='k', marker='+', s=150, linewidths=2)
    cbar = plt.colorbar(ha, ax=ax, shrink=0.83, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(f"Flux ({bunit})", rotation=270, fontsize=14, labelpad=20)
    styles = ['solid'] * len(nhi_levels)
    styles[0] = 'dashed'
    m0 = ax.contour(nhi, levels=nhi_levels, cmap='Oranges_r', norm='log', transform=ax.get_transform(mom0_wcs), linestyles=styles) # colors='white', cmap='bone'
    # m0.set_linestyles(styles)
    plt.xlim(0,ha_data.shape[1])
    plt.ylim(0,ha_data.shape[0])
    add_beam(ax, major=mom0_header['BMAJ'] * u.deg, minor=mom0_header['BMIN'] * u.deg, angle=mom0_header['BPA'], color='w')
    second_beam(ax, ha_wcs, seeing * u.arcsec, seeing * u.arcsec, 0)
    # add_beam(ax, major=seeing * u.arcsec, minor=seeing * u.arcsec, angle=0, color='k')
    add_scalebar(ax, scalebar_angle(d*u.Mpc, 5*u.kpc), label="5 kpc", color="w", label_top=True, corner="top left", fontproperties=FontProperties(size=14))

    ax.set_xlabel('RA', fontsize=16)
    ax.set_ylabel('Dec', fontsize=16)
    ax.tick_params(labelsize=14)

    plt.savefig(f"{out_dir}/ha_flux_hi_coldens_ngc{num}.pdf", dpi=300, bbox_inches='tight')

def main():
    args = get_args()
    in_path = args.in_path
    out_dir = args.out_dir
    num = out_dir.lower().split('ngc')[-1][:3]
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    plot_ha_flux_hi_coldens(in_path, num, out_dir)

if __name__ == "__main__":
    main()
