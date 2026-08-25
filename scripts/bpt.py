import os
import argparse
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True
from astropy.wcs import WCS
import matplotlib.patches as patches
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select output directory.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-e', '--elines', dest='elines_path', \
                        help='Path to ELINES file.')
    parser.add_argument('-r', '--reff', dest='re_path', \
                        help='Path to effective radius (re) file.')
    parser.add_argument('-m', '--mask', dest='mask_path', \
                        help='Path to mask.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    args = parser.parse_args()
    return args

def read_re_data(path, num):
    """
    for SFH data and Re
    """
    gal_path = path.replace('XXX', str(num))
    df = pd.read_csv(gal_path)
    ages = df['age (Gyr)'].values
    df = df.drop(columns=['age (Gyr)']) 
    df
    return ages, df

def calc_total_mass(re_data):
    mass_in = re_data.iloc[:,0].values
    mass_out = re_data.iloc[:,4].values
    mass = mass_in + mass_out
    total_mass = np.sum(mass_in) + np.sum(mass_out)
    return total_mass, mass

def plot_bpt(in_path, num, out_dir, mask_path=None, s2n=3):

    with fits.open(in_path) as hdul:
        head = hdul[0].header
        wcs = WCS(head).celestial
        oiii, e_oiii = hdul[0].data[1], hdul[0].data[121] # [OIII]5007
        hb, e_hb = hdul[0].data[3], hdul[0].data[123] # Hbeta
        ha, e_ha = hdul[0].data[20], hdul[0].data[140] # Halpha
        nii, e_nii = hdul[0].data[21], hdul[0].data[141] # [NII]6584
        sii, e_sii = hdul[0].data[24] + hdul[0].data[25], np.sqrt(hdul[0].data[144]**2 + hdul[0].data[145]**2) # [SII]6717+6731
        oi, e_oi = hdul[0].data[16], hdul[0].data[136] # [OI]6300

    if mask_path is not None:
        with fits.open(mask_path) as hdul:
            mask_re = hdul[0].data.astype(bool)

    oiii[oiii <= 0] = np.nan
    hb[hb <= 0] = np.nan
    nii[nii <= 0] = np.nan
    ha[ha <= 0] = np.nan
    sii[sii <= 0] = np.nan
    oi[oi <= 0] = np.nan

    mask1 = ((oiii / e_oiii > s2n) &
            (nii / e_nii > s2n) &
            (ha / e_ha > s2n) &
            (hb / e_hb > s2n))
    mask2 = ((oiii / e_oiii > s2n) &
            (sii / e_sii > s2n) &
            (ha / e_ha > s2n) &
            (hb / e_hb > s2n))
    mask3 = ((oiii / e_oiii > s2n) &
            (oi / e_oi > s2n) &
            (ha / e_ha > s2n) &
            (hb / e_hb > s2n))
    masks = [mask1, mask2, mask3]

    ob1 = np.log10(oiii[mask1] / hb[mask1])
    ob2 = np.log10(oiii[mask2] / hb[mask2])
    ob3 = np.log10(oiii[mask3] / hb[mask3])
    ob = [ob1, ob2, ob3]
    na = np.log10(nii[mask1] / ha[mask1])
    sa = np.log10(sii[mask2] / ha[mask2])
    oa = np.log10(oi[mask3] / ha[mask3])

    x_na_k01 = np.arange(-1.25, 0.47, 0.001)
    na_k01 = 0.61 / (x_na_k01 - 0.47) + 1.19
    x_na_k03 = np.arange(-1.25, 0.005, 0.001)
    na_k03 = 0.61 / (x_na_k03 - 0.05) + 1.3
    x_na_cf10 = np.arange(-0.199345, 0.5, 0.001) # -0.199345 intersection with na_k01
    na_cf10 = 1.01 * x_na_cf10 + 0.48 # empirical
    x_sa_k06 = np.arange(-1.25, 0.32, 0.001)
    sa_k06 = 0.72 / (x_sa_k06 - 0.32) + 1.3
    x_sa_k06_liner = np.arange(-0.314594, 0.5, 0.001) # -0.314594 intersection with sa_k06
    sa_k06_liner = 1.89 * x_sa_k06_liner + 0.76
    x_oa_k06 = np.arange(-1.25, 0.59, 0.001)
    oa_k06 = 0.73 / (x_oa_k06 - 0.59) + 1.33
    x_oa_k06_liner = np.arange(-0.52795, 0.5, 0.001) # -0.52795 intersection with oa_k06
    oa_k06_liner = 1.18 * x_oa_k06_liner + 1.30

    print(f"NGC{num} means: OIII/Hb = {np.mean(ob2):.3f}, OI/Ha = {np.mean(oa):.3f}, NII/Ha = {np.mean(na):.3f}, SII/Ha = {np.mean(sa):.3f}")
    print(f"NGC{num} integrated: OIII/Hb = {np.log10(np.nansum(oiii) / np.nansum(hb)):.3f}, OI/Ha = {np.log10(np.nansum(oi) / np.nansum(ha)):.3f}, NII/Ha = {np.log10(np.nansum(nii) / np.nansum(ha)):.3f}, SII/Ha = {np.log10(np.nansum(sii) / np.nansum(ha)):.3f}")

    fig, ax = plt.subplots(ncols=3, figsize=(15,5))
    for i, a in enumerate(ax):
        x = [na, sa, oa][i]
        if mask_path is not None:
            mask_in = mask_re[masks[i]]
            if num == '833':
                a.scatter(x[~mask_in], ob[i][~mask_in], s=10, color='r', label='R > HLR')
                a.scatter(x[mask_in], ob[i][mask_in], s=10, color='k', alpha=0.2, label='R < HLR')
            else:
                a.scatter(x[~mask_in], ob[i][~mask_in], s=3, color='r', label='R > HLR')
                a.scatter(x[mask_in], ob[i][mask_in], s=3, color='k', alpha=0.2, label='R < HLR')
        else:
            a.scatter(x, ob[i], s=1, color='k')
        a.scatter(np.mean(x), np.mean(ob[1]), color='orange', marker='+', s=150, linewidths=4)
        a.set_xlim(-1.0, 0.5)
        a.set_ylim(-1.25, 0.8)
        if i == 0:
            a.plot(x_na_k01, na_k01, label='Kewley+01', color='C0')
            a.plot(x_na_k03, na_k03, linestyle='-.', label='Kauffmann+03', color='C1')
            a.plot(x_na_cf10, na_cf10, linestyle='--', label='Cid Fernandes+10', color='C5')
            a.set_xlabel(r'$\log\left(\frac{[NII]6584}{H\alpha}\right)$')
            a.set_ylabel(r'$\log\left(\frac{[OIII]5007}{H\beta}\right)$')
        elif i == 1:
            a.plot(x_sa_k06, sa_k06, label='Kewley+06', color='C0')
            a.plot(x_sa_k06_liner, sa_k06_liner, linestyle='--', label='Kewley+06 LINER/SF', color='C5')
            a.set_xlabel(r'$\log\left(\frac{[SII](6717+6731)}{H\alpha}\right)$')
        else:
            a.plot(x_oa_k06, oa_k06, label='Kewley+06', color='C0')
            a.plot(x_oa_k06_liner, oa_k06_liner, linestyle='--', label='Kewley+06 LINER/SF', color='C5')
            a.set_xlabel(r'$\log\left(\frac{[OI]6300}{H\alpha}\right)$')
            a.set_xlim(-2.5, 0.5)

    plt.savefig(f"{out_dir}/bpt_full_ngc{num}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_dir}/bpt_full_ngc{num}.png", dpi=300, bbox_inches='tight')

def plot_bpt_whan_sfh(elines_path, re_path, num, out_dir, mask_path=None, s2n=3):
    with fits.open(elines_path) as hdul:
        head = hdul[0].header
        wcs = WCS(head).celestial
        oiii, e_oiii = hdul[0].data[1], hdul[0].data[121] # [OIII]5007
        hb, e_hb = hdul[0].data[3], hdul[0].data[123] # Hbeta
        ha, e_ha = hdul[0].data[20], hdul[0].data[140] # Halpha
        nii, e_nii = hdul[0].data[21], hdul[0].data[141] # [NII]6584
        ew_ha, e_ew_ha = hdul[0].data[110], hdul[0].data[230] # EW(Halpha)
        ew_ha = np.abs(ew_ha)
    
    ages, re_data = read_re_data(re_path, num)
    total_mass, mass = calc_total_mass(re_data)
    
    if mask_path is not None:
        with fits.open(mask_path) as hdul:
            mask_re = hdul[0].data.astype(bool)

    fig, ax = plt.subplots(ncols=3, figsize=(18,5))

    bpt(ax[0], oiii, e_oiii, hb, e_hb, ha, e_ha, nii, e_nii, mask_re=mask_re, s2n=s2n)
    whan(ax[1], ew_ha, e_ew_ha, nii, e_nii, ha, e_ha, mask_re=mask_re, s2n=s2n)
    sfh(ax[2], ages, re_data, norm=mass)

    fig.savefig(f"{out_dir}/bpt_whan_sfh_ngc{num}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(f"{out_dir}/bpt_whan_sfh_ngc{num}.png", dpi=300, bbox_inches='tight')

def bpt(ax, oiii, e_oiii, hb, e_hb, ha, e_ha, nii, e_nii, mask_re=None, s2n=3):
    oiii[oiii <= 0] = np.nan
    hb[hb <= 0] = np.nan
    nii[nii <= 0] = np.nan
    ha[ha <= 0] = np.nan

    mask = ((oiii / e_oiii > s2n) &
            (nii / e_nii > s2n) &
            (ha / e_ha > s2n) &
            (hb / e_hb > s2n))
    
    ob = np.log10(oiii[mask] / hb[mask])
    na = np.log10(nii[mask] / ha[mask])

    x_na_k01 = np.arange(-1.25, 0.47, 0.001)
    na_k01 = 0.61 / (x_na_k01 - 0.47) + 1.19
    x_na_k03 = np.arange(-1.25, 0.005, 0.001)
    na_k03 = 0.61 / (x_na_k03 - 0.05) + 1.3
    x_na_cf10 = np.arange(-0.199345, 0.5, 0.001) # -0.199345 intersection with na_k01
    na_cf10 = 1.01 * x_na_cf10 + 0.48 # empirical

    if mask_re is not None:
        mask_in = mask_re[mask]
        ax.scatter(na[~mask_in], ob[~mask_in], s=3, color='r') # , label='R > HLR'
        ax.scatter(na[mask_in], ob[mask_in], s=3, color='k') # , label='R < HLR'
    else:
        ax.scatter(na, ob, s=1, color='k')
    
    ax.scatter(np.mean(na), np.mean(ob), color='orange', marker='+', s=100, linewidths=3)
    ax.set_xlim(-1.0, 0.5)
    ax.set_ylim(-1.25, 0.8)
    ax.plot(x_na_k01, na_k01, label='Kewley+01', color='C0')
    ax.plot(x_na_k03, na_k03, linestyle='-.', label='Kauffmann+03', color='C1')
    ax.plot(x_na_cf10, na_cf10, linestyle='--', label='Cid Fernandes+10', color='C5')
    ax.set_xlabel(r'$\log\left(\frac{[NII]6584}{H\alpha}\right)$')
    ax.set_ylabel(r'$\log\left(\frac{[OIII]5007}{H\beta}\right)$')
    ax.legend()

def whan(ax, ew_ha, e_ew_ha, nii, e_nii, ha, e_ha, mask_re=None, s2n=3):
    ew_ha[ew_ha == 0] = np.nan
    nii[nii <= 0] = np.nan
    ha[ha <= 0] = np.nan

    mask = ((ew_ha / e_ew_ha > s2n) &
                (nii / e_nii > s2n) &
                (ha / e_ha > s2n))

    ew = ew_ha[mask]
    na = np.log10(nii[mask] / ha[mask])

    if mask_re is not None:
        mask_in = mask_re[mask]
        ax.scatter(na[~mask_in], ew[~mask_in], s=3, color='r', label='R > HLR')
        ax.scatter(na[mask_in], ew[mask_in], s=3, color='k', label='R < HLR')
    else:
        ax.scatter(na, ew, s=1, color='k')
    
    ax.set_xlabel(r'$\log\left(\frac{[NII]6584}{H\alpha}\right)$')
    ax.set_ylabel(r'$EW(H\alpha)$ ($\AA$)')
    ax.legend()
    # ax.scatter(np.mean(na), np.mean(ew), color='orange', marker='+', s=100, linewidths=3)
    ax.set_xlim(-1, 0.6)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 400)
    ax.axvline(-0.4, )
    ax.axhline(0.5, )
    ax.axhline(3, )
    ax.axhline(6, )

def sfh(ax, ages, re_data, norm=None):
    dex_per_bin = np.median(np.diff(np.log10(ages)))
    sigma_pixels = 0.3 / dex_per_bin

    sfh_in = re_data.iloc[:,2].values
    sfh_out = re_data.iloc[:,6].values
    conv_in = gaussian_filter1d(sfh_in, sigma=sigma_pixels)
    conv_out = gaussian_filter1d(sfh_out, sigma=sigma_pixels)
    conv_out = conv_out * (conv_in[0]/conv_out[0])
    if norm is not None:
        conv_in = conv_in * 1e11 / np.cumsum(norm[::-1])[::-1]
        conv_out = conv_out * 1e11 / np.cumsum(norm[::-1])[::-1]
        conv_in = np.log10(conv_in)
        conv_out = np.log10(conv_out)
        no = 'linear'
    else:
        no = 'log'

    # rat = conv_in[0] / conv_out[0]
    # print(rat)
    # +(conv_in[0]-conv_out[0])

    ax.plot(ages, conv_in, label=f'$R < 1$ HLR', c='k')
    ax.plot(ages, conv_out, label=f'$R > 1$ HLR', linestyle='--', c='r')
    ax.set_ylim(-0.8,3)
    ax.set_ylabel(r'Normalized $\log sSFR$ (10$^{11}$ yr$^{-1}$)', fontsize=12)
    ax.set_yscale(no)
    ax.set_xlabel('Lookback time (Gyr)', fontsize=12)
    ax.set_xscale('log')
    ax.legend()

def main():
    args = get_args()
    elines_path = args.elines_path
    re_path = args.re_path
    mask_path = args.mask_path
    out_dir = args.out_dir
    num = out_dir.lower().split('ngc')[-1][:3]
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    plot_bpt_whan_sfh(elines_path, re_path, num, out_dir, mask_path=mask_path, s2n=3)
    plot_bpt(elines_path, num, out_dir, mask_path, 3)

if __name__ == "__main__":
    main()