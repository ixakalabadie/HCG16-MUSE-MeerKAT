import os
import argparse
import configparser
import json
from astropy.io import fits
import astropy.units as u
from astropy.convolution import convolve, Gaussian2DKernel, Gaussian1DKernel
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True

LAB = {'833': 'B', '835': 'A', '838': 'C', '839': 'D', '848': '848'}

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-r', '--radius', dest='radius_path', \
                        help='Path to radius SFH data.')
    parser.add_argument('-f', '--sfh', dest='sfh_path', \
                        help='Path to SFH data.')
    parser.add_argument('-e', '--eff', dest='re_path', \
                        help='Path to Re data.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders may be created there.')
    parser.add_argument('-g', '--galaxies', dest='galaxies', \
                        help='List of numbers of galaxies.', nargs='+', type=int)
    args = parser.parse_args()
    return args

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

def read_radius_data(radius_path, galaxies, mass=True, hlr=False):
    data = []
    for gal in galaxies:
        if mass:
            if hlr:
                gal_path = radius_path.replace('XXX', 'mass_HLR')
            else:
                gal_path = radius_path.replace('XXX', 'mass')
        else:
            if hlr:
                gal_path = radius_path.replace('XXX', 'sfh_HLR')
            else:
                gal_path = radius_path.replace('XXX', 'sfh')
        gal_path = gal_path.replace('YYY', str(gal))
        print(gal_path)
        df = pd.read_csv(gal_path)  
        ages = df['age (Gyr)'].values
        print(ages)
        df = df.drop(columns=['age (Gyr)']) 
        data.append(df)
    if mass:
        radii = [col[len('M_'):-len(' (M_sun)')] for col in df.columns if col.startswith('M_')]
    else:
        radii = [col[len('SFR_'):-len(' (M_sun/yr)')] for col in df.columns if col.startswith('SFR_')]
    return data, ages, radii

def read_sfh_data(sfh_path, galaxies):
    """
    for SFH data and Re
    """
    data = []
    for gal in galaxies:
        gal_path = sfh_path.replace('XXX', str(gal))
        df = pd.read_csv(gal_path)
        df = df.drop(columns=['age (Gyr)']) 
        data.append(df)
    return data

def plot_radius(ages, radius_data, radius, galaxies, total_masses, out_dir, ylim=(1e-4, 9e2)):
    markers = ['o', 'x', '<', 'd', '+', '*', 's']
    sort_idx = np.argsort(total_masses)[::-1]
    # Colormap from red to blue
    cmap = plt.cm.coolwarm_r  # or plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(galaxies)))[sort_idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f'SFH at R={radius} arcsec')
    for i, gal in enumerate(galaxies):
        sfh = radius_data[i][f'SFR_{radius} (M_sun/yr)'].values
        sfh_err = radius_data[i][f'DSFR_{radius} (M_sun/yr)'].values
        # ax.errorbar(ages, sfh, yerr=sfh_err, label=f'NGC{gal}', fmt='.', ecolor='k', capsize=3)
        if gal == 848 or gal == '848':
            ax.scatter(ages, sfh, marker=markers[i], label=f'NGC {gal}', color=colors[i], s=45)
        else:
            ax.scatter(ages, sfh, marker=markers[i], label=f'HCG 16{LAB[str(gal)]}', color=colors[i], s=45)
    ax.set_yscale('log')
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xscale('log')
    ax.set_xlabel('Lookback time (Gyr)', fontsize=16)
    ax.set_ylabel(r'SFR (M$_\odot$/yr)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend() # fontsize=12 is too much

    plt.savefig(f"{out_dir}/images/comparison_radius_sfh_R{radius}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/images/comparison_radius_sfh_R{radius}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_radius_together(ages, radius_data, radii, galaxies, out_dir, ylim=(2e-4, 9e2)):
    """
    Plot SFR-age scatter plots for all galaxies at a single radius.
    """
    try:
        markers = ['o', 'x', '<', 'd', '+', '*', 's']
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        for j, ra in enumerate(radii):
            for i, gal in enumerate(galaxies):
                sfh = radius_data[i][f'SFR_{ra} (M_sun/yr)'].values
                sfh_err = radius_data[i][f'DSFR_{ra} (M_sun/yr)'].values
                if j < 3:
                    row = 0
                else:
                    row = 1
                if gal == 848 or gal == '848':
                    ax[row, j-3*row].scatter(ages, sfh, marker=markers[i], label=f'NGC {gal}')
                else:
                    ax[row, j-3*row].scatter(ages, sfh, marker=markers[i], label=f'HCG 16{LAB[str(gal)]}')
                
                ax[row, j-3*row].set_yscale('log')
                ax[row, j-3*row].set_ylim(ylim[0], ylim[1])
                ax[row, j-3*row].set_xscale('log')
                ax[row, j-3*row].tick_params(axis='both', labelsize=14)
                ax[row, j-3*row].text(0.67, 0.93, f'$R={float(ra):.0f}$ arcsec', transform=ax[row, j-3*row].transAxes)
                if j < 3:
                    ax[row, j-3*row].set_xticks([])
                else:
                    ax[row, j-3*row].set_xlabel('Lookback time (Gyr)', fontsize=16)
                if j % 3 == 0:
                    ax[row, j-3*row].set_ylabel(r'SFR (M$_\odot$/yr)', fontsize=16)
                else:
                    ax[row, j-3*row].set_yticks([])

        ax[1,0].legend(fontsize=12)
        fig.savefig(f"{out_dir}/images/comparison_radius_sfh_allradii.pdf", dpi=300, bbox_inches='tight')
        fig.savefig(f"{out_dir}/images/comparison_radius_sfh_allradii.png", dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass

def plot_mass_accumulation(ages, re_data, galaxies, total_masses, out_dir, norm=False):
    """
    Accumulated mass as a function of lookback time, for all galaxies in the same plot, with solid lines for inside HLR and dashed lines for outside HLR.
    """
    sort_idx = np.argsort(total_masses)[::-1]
    # Colormap from red to blue
    cmap = plt.cm.coolwarm_r  # or plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(galaxies)))[sort_idx]

    fig, ax = plt.subplots(figsize=(6,6))
    for i, gal in enumerate(galaxies):
        mass_in = re_data[i].iloc[:,0].values
        mass_out = re_data[i].iloc[:,4].values
        tot_in = np.cumsum(mass_in[::-1])[::-1]
        tot_out = np.cumsum(mass_out[::-1])[::-1]
        if norm:
            tot_in = tot_in / tot_in[0]
            tot_out = tot_out / tot_out[0]
        lab = f'$\\log M_{{\\star, {LAB[str(gal)]}}}$ = {np.log10(total_masses[i]):.2f} M$_\\odot$'
        inner, = ax.plot(ages, tot_in, label=lab, color=colors[i])
        ax.plot(ages, tot_out, '--', color=colors[i])
    
    ax.set_xlabel('Lookback time (Gyr)', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xscale('log')
    plt.legend(fontsize=14)

    if norm:
        ax.set_ylabel('Normalized accumulated mass', fontsize=16)
        plt.savefig(f"{out_dir}/images/mass_accumulation_Re_norm.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{out_dir}/images/mass_accumulation_Re_norm.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        ax.set_yscale('log')
        ax.set_ylabel(r'Accumulated mass (M$_\odot$)', fontsize=16)
        plt.savefig(f"{out_dir}/images/mass_accumulation_Re.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{out_dir}/images/mass_accumulation_Re.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_convolved(ages, re_data, galaxies, total_masses, out_dir, norm=None):
    sort_idx = np.argsort(total_masses)[::-1]
    # Colormap from red to blue
    cmap = plt.cm.coolwarm_r  # or plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(galaxies)))[sort_idx]

    dex_per_bin = np.median(np.diff(np.log10(ages)))
    for dex in [ 0.3]:
        sigma_pixels = dex / dex_per_bin

        fig, ax = plt.subplots(figsize=(6,6))
        for i, gal in enumerate(galaxies):
            sfh_in = re_data[i].iloc[:,2].values
            sfh_out = re_data[i].iloc[:,6].values
            conv_in = gaussian_filter1d(sfh_in, sigma=sigma_pixels)
            conv_out = gaussian_filter1d(sfh_out, sigma=sigma_pixels)
            if norm is not None:
                conv_in = conv_in * 1e11 / np.cumsum(norm[i][::-1])[::-1]
                conv_out = conv_out * 1e11 / np.cumsum(norm[i][::-1])[::-1]
                conv_in = np.log10(conv_in)
                conv_out = np.log10(conv_out)
                no = 'linear'
            else:
                no = 'log'
            ax.plot(ages, conv_in, label=f'$SFH_{{{gal}}} < 1 HLR$', color=colors[i])
            ax.plot(ages, conv_out, '--', color=colors[i])
        
        ax.set_xlabel('Lookback time (Gyr)')
        ax.set_xscale('log')
        plt.legend(fontsize=12)
        if norm:
            ax.set_ylabel(r'$\log sSFR$ (10$^{-11}$ yr$^{-1}$)', fontsize=16)
            plt.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}_specific.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}_specific.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            ax.set_yscale('log')
            ax.set_ylabel(r'SFR (M$_\odot$/yr)')
            fig.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}.pdf", dpi=300, bbox_inches='tight')
            fig.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}.png", dpi=300, bbox_inches='tight')
            plt.close()

def plot_convolved2(ages, re_data, galaxies, total_masses, out_dir, norm=None):
    """
    Convolved SFH for all galaxies in the same plot, with solid lines for inside HLR and dashed lines for outside HLR.
    """
    sort_idx = np.argsort(total_masses)[::-1]
    # Colormap from red to blue
    cmap = plt.cm.coolwarm_r  # or plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(galaxies)))[sort_idx]

    dex_per_bin = np.median(np.diff(np.log10(ages)))
    for dex in [0.25]: # In the others is 0.25
        sigma_pixels = dex / dex_per_bin

        fig, ax = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
        plt.subplots_adjust(wspace=0.0)
        for i, gal in enumerate(galaxies):
            sfh_in = re_data[i].iloc[:,2].values
            sfh_out = re_data[i].iloc[:,6].values
            conv_in = gaussian_filter1d(sfh_in, sigma=sigma_pixels)
            conv_out = gaussian_filter1d(sfh_out, sigma=sigma_pixels)
            if norm is not None:
                conv_in = conv_in * 1e11 / np.cumsum(norm[i][::-1])[::-1]
                conv_out = conv_out * 1e11 / np.cumsum(norm[i][::-1])[::-1]
                conv_in = np.log10(conv_in)
                conv_out = np.log10(conv_out)
                no = 'linear'
            else:
                no = 'log'
            ax[0].plot(ages, conv_in, label=f'$R_{{{LAB[str(gal)]}}} < 1$ HLR', color=colors[i])
            ax[1].plot(ages, conv_out, label=f'$R_{{{LAB[str(gal)]}}} > 1$ HLR', color=colors[i])
        
        ax[0].set_xlabel('Lookback time (Gyr)', fontsize=16)
        ax[0].set_xscale('log')
        ax[0].legend(fontsize=12)
        ax[1].set_xlabel('Lookback time (Gyr)', fontsize=16)
        ax[1].set_xscale('log')
        ax[1].legend(fontsize=12)
        if norm:
            ax[0].set_ylabel(r'$\log sSFR$ (10$^{-11}$ yr$^{-1}$)', fontsize=16)
            # ax[1].set_ylabel(r'$\log sSFR$ (10$^{-11}$ yr$^{-1}$)', fontsize=12)
            plt.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}_specific.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}_specific.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            ax[0].set_ylabel(r'SFR (M$_\odot$/yr)', fontsize=16)
            ax[1].set_ylabel(r'SFR (M$_\odot$/yr)', fontsize=16)
            fig.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}.pdf", dpi=300, bbox_inches='tight')
            fig.savefig(f"{out_dir}/images/convolved_SFH_HLR_{dex}.png", dpi=300, bbox_inches='tight')
            plt.close()

def calc_total_mass(re_data, galaxies):
    total_masses = []
    masses = []
    for i, gal in enumerate(galaxies):
        mass_in = re_data[i].iloc[:,0].values
        mass_out = re_data[i].iloc[:,4].values
        masses.append(mass_in + mass_out)
        total_masses.append(np.sum(mass_in) + np.sum(mass_out))
    
    return total_masses, masses

def plot_radius_all(ages, radius_data, radii, galaxies, total_masses, out_dir, norm=None, mass=False, vmin=0.001, vmax=100, conv=False, hlr=False):
    """
    Plot radius-age 2D images
    """
    sort_idx = np.argsort(total_masses)[::-1]
    log_ages = np.log10(ages)
    # midpoints between consecutive ages in log space
    log_edges = np.concatenate([
        [log_ages[0] - (log_ages[1] - log_ages[0]) / 2],
        (log_ages[:-1] + log_ages[1:]) / 2,
        [log_ages[-1] + (log_ages[-1] - log_ages[-2]) / 2]
    ])
    edges = 10**log_edges
    y_edges = np.arange(len(radii) + 1) - 0.5

    fig, ax = plt.subplots(nrows=len(galaxies), figsize=(10,14))
    plt.subplots_adjust(hspace=0.06)
    for i, gal in enumerate(galaxies):
        sfh = radius_data[i].iloc[:,::2].values.T
        sfh[sfh <= 0] = np.nan
        # sfh shape is (num_radii, num_ages)
        con = ''
        if conv:
            dex_per_bin = np.median(np.diff(np.log10(ages)))
            sigma_pixels = 0.25 / dex_per_bin
            sigma_sfh = 2
            kernel = Gaussian2DKernel(x_stddev=sigma_pixels, y_stddev=sigma_sfh)
            sfh = convolve(sfh, kernel, preserve_nan=False)
            con = '_convolved'
        if norm is not None:
            # sfh = sfh * 1e11 / total_masses[i]
            sfh = sfh * 1e11 / np.cumsum(norm[i][::-1])[::-1] # 1e11 typical unit. necessary for cbar.
            sfh = np.log10(sfh)
            lab = '_specific'
            no = 'linear'
        else:
            lab = ''
            no = 'log'
        im = ax[sort_idx[i]].pcolormesh(edges, y_edges, sfh, norm=no, vmin=vmin, vmax=vmax, cmap='jet')
        # Add vertical lines at 0.06, 0.4 and 1 Gyr
        ax[sort_idx[i]].axvline(6e-2, c='k')
        ax[sort_idx[i]].axvline(0.5, c='k')
        ax[sort_idx[i]].axvline(1, c='k')
        if len(radii) < 10:
            ax[sort_idx[i]].set_yticks(y_edges, labels=['0'] + [f"{r}" for r in radii]);
        else:
            ax[sort_idx[i]].set_yticks(y_edges[[0,10,20,30]], labels=['0', '1', '2', '3']);
        if hlr:
            ax[sort_idx[i]].set_ylabel(r'$R_{%s}$ (HLR)'%LAB[str(gal)], fontsize=16)
        else:
            ax[sort_idx[i]].set_ylabel(r'$R_{%s}$ (arcsec)'%LAB[str(gal)], fontsize=16)
        ax[sort_idx[i]].set_aspect('auto')
        ax[sort_idx[i]].tick_params(axis='both', labelsize=14)
        ax[sort_idx[i]].set_xscale('log')
        if i == len(galaxies)-1:
            ax[sort_idx[i]].set_xlabel('Lookback time (Gyr)', fontsize=16)
        else:
            ax[sort_idx[i]].set_xticks([])

    cbar = fig.colorbar(im, ax=ax, location='top', fraction=0.035, pad=0.005, aspect=30);
    cbar.ax.tick_params(labelsize=14)
    if mass:
        cbar.ax.set_title(r'M (M$_\odot$)', fontsize=16)
        plt.savefig(f'{out_dir}/images/comparison_radial_mass_all{lab}{con}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_dir}/images/comparison_radial_mass_all{lab}{con}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        if norm is not None:
            cbar.ax.set_title(r'$\log sSFR$ (10$^{-11}$ yr$^{-1}$)', fontsize=16)
        else:
            cbar.ax.set_title(r'SFR (M$_\odot$/yr)', fontsize=16)
        plt.savefig(f'{out_dir}/images/comparison_radial_sfh_all{lab}{con}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{out_dir}/images/comparison_radial_sfh_all{lab}{con}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_sfh(ages, sfh_data, galaxies, total_masses, out_dir, norm=None, conv=False):
    sort_idx = np.argsort(total_masses)[::-1]
    # Colormap from red to blue
    cmap = plt.cm.coolwarm_r  # or plt.cm.coolwarm
    colors = cmap(np.linspace(0, 1, len(galaxies)))[sort_idx]
    fig, ax = plt.subplots(figsize=(14,5))
    for i, gal in enumerate(galaxies):
        sfh = sfh_data[i]['SFR (M_sun/yr)']
        sfh_err = sfh_data[i]['DSFR (M_sun/yr)']
        con = ''
        if conv:
            dex_per_bin = np.median(np.diff(np.log10(ages)))
            sigma_pixels = 0.25 / dex_per_bin
            sigma_sfh = 2
            kernel = Gaussian1DKernel(sigma_pixels)
            sfh = convolve(sfh, kernel, preserve_nan=False)
            con = '_convolved'
        if norm is not None:
            sfh = sfh * 1e11 / np.cumsum(norm[i][::-1])[::-1]
            sfh_err = sfh_err * 1e11 / np.cumsum(norm[i][::-1])[::-1]
            sfh_err = sfh_err / sfh / np.log(10) # convert to log error
            sfh = np.log10(sfh)
            lab = '_specific'
        else:
            lab = ''
        if gal == 848 or gal == '848':
            ax.errorbar(ages, sfh, yerr=sfh_err, label=f'NGC {gal}', fmt='o-', color=colors[i], ecolor='k', capsize=3)
        else:
            ax.errorbar(ages, sfh, yerr=sfh_err, label=f'HCG 16{LAB[str(gal)]}', fmt='o-', color=colors[i], ecolor='k', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Lookback time (Gyr)', fontsize=16)
    if norm is not None:
        ax.set_ylabel(r'$\log sSFR$ (10$^{-11}$ yr$^{-1}$)', fontsize=16)
    else:
        ax.set_xscale('log')
        ax.set_ylabel(r'SFR (M$_\odot$/yr)', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(f"{out_dir}/images/comparison_SFH_linear{lab}{con}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/images/comparison_SFH_linear{lab}{con}.png", dpi=300, bbox_inches='tight')

    ax.set_yscale('log')
    plt.savefig(f"{out_dir}/images/comparison_SFH{lab}{con}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/images/comparison_SFH{lab}{con}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_mass(ages, data, galaxies, out_dir, norm=None):
    """
    norm = total mass for each galaxy
    """
    fig, ax = plt.subplots(figsize=(14,5))
    for i, gal in enumerate(galaxies):
        sfh = data[i]['M (M_sun)']
        sfh_err = data[i]['DM (M_sun)']
        if norm is not None:
            sfh = sfh / np.cumsum(norm[i][::-1])[::-1]
            sfh_err = sfh_err / np.cumsum(norm[i][::-1])[::-1]
            # sfh_err = sfh_err / sfh / np.log(10) # convert to log error
            # sfh = np.log10(sfh)
            lab = '_specific'
        else:
            lab = ''
        if gal == 848 or gal == '848':
            ax.errorbar(ages, sfh, yerr=sfh_err, label=f'NGC {gal}', fmt='o-', ecolor='k', capsize=3)
        else:
            ax.errorbar(ages, sfh, yerr=sfh_err, label=f'HCG 16{LAB[str(gal)]}', fmt='o-', ecolor='k', capsize=3)
        
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Lookback time (Gyr)', fontsize=16)
    if norm is not None:
        ax.set_ylabel(r'$\log sSFR$ (10$^{-11}$ yr$^{-1}$)', fontsize=16)
    else:
        ax.set_ylabel(r'SFR (M$_\odot$/yr)', fontsize=16)
    ax.legend(fontsize=12)

    plt.savefig(f"{out_dir}/images/comparison_mass_linear{lab}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/images/comparison_mass_linear{lab}.png", dpi=300, bbox_inches='tight')

    ax.set_yscale('log')
    plt.savefig(f"{out_dir}/images/comparison_mass{lab}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/images/comparison_mass{lab}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    args = get_args()
    radius_path = args.radius_path # with xxx instead of sfh/mass and yyy instead of galaxy number
    sfh_path = args.sfh_path # with xxx instead of galaxy number
    re_path = args.re_path # with xxx instead of galaxy number
    galaxies = args.galaxies
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(f'{out_dir}/images'):
        os.makedirs(f'{out_dir}/images')

    # Our set radii
    # Read data:
    radius_sfh_data, ages, radii = read_radius_data(radius_path, galaxies, mass=False)
    radius_mass_data, _, _, = read_radius_data(radius_path, galaxies, mass=True)
    sfh_data = read_sfh_data(sfh_path, galaxies)
    re_data = read_sfh_data(re_path, galaxies)
    total_masses, masses = calc_total_mass(re_data, galaxies)

    # Integrated SFH plots:
    plot_mass_accumulation(ages, re_data, galaxies, total_masses, out_dir, norm=True)
    plot_mass_accumulation(ages, re_data, galaxies, total_masses, out_dir, norm=False)
    plot_sfh(ages, sfh_data, galaxies, total_masses, out_dir)
    plot_sfh(ages, sfh_data, galaxies, total_masses, out_dir, norm=masses)
    plot_sfh(ages, sfh_data, galaxies, total_masses, out_dir, conv=True)
    plot_sfh(ages, sfh_data, galaxies, total_masses, out_dir, norm=masses, conv=True)
    plot_mass(ages, sfh_data, galaxies, out_dir)
    plot_mass(ages, sfh_data, galaxies, out_dir, norm=masses)
    plot_convolved2(ages, re_data, galaxies, total_masses, out_dir)
    plot_convolved2(ages, re_data, galaxies, total_masses, out_dir, norm=masses)

    # Radial SFH plots:
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, vmin=0.001, vmax=80)
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, norm=masses, vmin=-2.3, vmax=2.3)
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, norm=masses, conv=True, vmin=-2.3, vmax=2.3)
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, conv=True, vmin=0.001, vmax=80)
    plot_radius_all(ages, radius_mass_data, radii, galaxies, total_masses, out_dir, mass=True, vmin=1e5, vmax=3e9)
    plot_radius_all(ages, radius_mass_data, radii, galaxies, total_masses, out_dir, mass=True, conv=True, vmin=1e5, vmax=3e9)

    # SFR for each radius scatter plot:
    plot_radius_together(ages, radius_sfh_data, radii, galaxies, out_dir)
    for r in radii:
        plot_radius(ages, radius_sfh_data, r, galaxies, total_masses, out_dir)

    # HLR radii
    radius_sfh_data, ages, radii = read_radius_data(radius_path, galaxies, mass=False, hlr=True)
    radius_mass_data, _, _, = read_radius_data(radius_path, galaxies, mass=True, hlr=True)
    
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, vmin=0.001, vmax=80, hlr=True)
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, norm=masses, vmin=-2.3, vmax=2.3, hlr=True)
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, norm=masses, conv=True, vmin=-2.3, vmax=2.3, hlr=True)
    plot_radius_all(ages, radius_sfh_data, radii, galaxies, total_masses, out_dir, conv=True, vmin=0.001, vmax=80, hlr=True)
    plot_radius_all(ages, radius_mass_data, radii, galaxies, total_masses, out_dir, mass=True, vmin=1e5, vmax=3e9, hlr=True)
    plot_radius_all(ages, radius_mass_data, radii, galaxies, total_masses, out_dir, mass=True, conv=True, vmin=1e5, vmax=3e9, hlr=True)

if __name__ == "__main__":
    main()