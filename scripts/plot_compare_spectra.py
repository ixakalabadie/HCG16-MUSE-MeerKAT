# Code created with the assistance of Claude
import os
import re
import math
import time
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.convolution import Box1DKernel, convolve
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
plt.rcParams['axes.formatter.use_mathtext'] = True

C_KMS, F0 = 299792.458, 1.420405751e9
COMMON_RES = 20.0

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select input parameters.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-ch", "--cube-high", dest="cube_high", required=True, help="high-resolution cube (FITS)")
    parser.add_argument("-cl", "--cube-low", dest="cube_low", required=True, help="low-resolution cube (FITS)")
    parser.add_argument("-mh", "--mask-high", dest="mask_high", required=True, help="high-resolution mask (FITS)")
    parser.add_argument("-ml", "--mask-low", dest="mask_low", required=True, help="low-resolution mask (FITS)")
    parser.add_argument("-n", "--areas-name", dest="gal_name", help="Path to galaxy files")
    parser.add_argument('-g', '--galaxies', dest='galaxies', \
                        help='List of numbers of galaxies.', nargs='+', type=int)
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                    help='Outup directory. New folders may be created there.')
    args = parser.parse_args()
    return args

def load_cube(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        data = np.squeeze(data)
        header = hdul[0].header
        wcs = WCS(header)
    return data, header, wcs

def cube_extrema(wcs, header, rest_freq=1.420405751e9):
    """Print min/max of RA, Dec and the spectral axis (pixel edges)."""
    n = {i: header[f"NAXIS{i+1}"] for i in range(wcs.naxis)}   # FITS axis lengths
    nx, ny = n[wcs.wcs.lng], n[wcs.wcs.lat]

    # Sky: sample all four image edges, because the projection is curved
    t = np.linspace(-0.5, nx - 0.5, 400); u = np.linspace(-0.5, ny - 0.5, 400)
    xe = np.r_[t, t, np.full_like(u, -0.5), np.full_like(u, nx - 0.5)]
    ye = np.r_[np.full_like(t, -0.5), np.full_like(t, ny - 0.5), u, u]
    ra, dec = wcs.celestial.wcs_pix2world(xe, ye, 0)
    ra = (ra - ra[0] + 180) % 360 + ra[0] - 180                 # avoid the 0/360 wrap
    print(f"  RA : {ra.min():.5f} to {ra.max():.5f} deg")
    print(f"  Dec: {dec.min():.5f} to {dec.max():.5f} deg")

    # Spectral axis
    ispec = wcs.wcs.spec
    nchan = n[ispec]
    v = np.sort(np.asarray(wcs.spectral.pixel_to_world_values(np.array([-0.5, nchan - 0.5])), float))
    if wcs.wcs.ctype[ispec].startswith("FREQ"):
        print(f"  Freq: {v[0]/1e6:.4f} to {v[1]/1e6:.4f} MHz")
        vel = 299792.458 * (rest_freq - v) / rest_freq
        print(f"  Vel (radio, HI): {vel.min():.1f} to {vel.max():.1f} km/s")
    else:
        print(f"  Vel: {v[0]/1e3:.1f} to {v[1]/1e3:.1f} km/s")

def footprint_in_pixels(wcs_o, hdr_o, wcs_k):
    """Footprint of cube o (sky edges and spectral edges) in the pixel grid of cube k."""
    n = {i: hdr_o[f"NAXIS{i+1}"] for i in range(wcs_o.naxis)}
    nx, ny, nc = n[wcs_o.wcs.lng], n[wcs_o.wcs.lat], n[wcs_o.wcs.spec]
    t, u = np.linspace(-0.5, nx - 0.5, 400), np.linspace(-0.5, ny - 0.5, 400)
    xe = np.r_[t, t, np.full_like(u, -0.5), np.full_like(u, nx - 0.5)]
    ye = np.r_[np.full_like(t, -0.5), np.full_like(t, ny - 0.5), u, u]
    ra, dec = wcs_o.celestial.wcs_pix2world(xe, ye, 0)
    x, y = wcs_k.celestial.wcs_world2pix(ra, dec, 0)
    s = np.asarray(wcs_o.spectral.pixel_to_world_values(np.array([-0.5, nc - 0.5])), float).ravel()
    c = np.asarray(wcs_k.spectral.world_to_pixel_values(s), float).ravel()
    return x, y, c

def make_subcube(data, header, wcs, other_header, other_wcs):
    """Subcube of `data` covering the volume shared with the other cube (cut at pixel edges)."""
    N = wcs.naxis
    n = {i: header[f"NAXIS{i+1}"] for i in range(N)}
    lng, lat, spec = wcs.wcs.lng, wcs.wcs.lat, wcs.wcs.spec
    x, y, c = footprint_in_pixels(other_wcs, other_header, wcs)
    def rng(p, L):      # pixel-edge coordinates, rounded to the nearest whole pixel
        return int(np.round(max(p.min() + 0.5, 0))), int(np.round(min(p.max() + 0.5, L)))
    sl = [slice(None)] * N                                 # array order is reversed
    sl[N-1-lng]  = slice(*rng(x, n[lng]))
    sl[N-1-lat]  = slice(*rng(y, n[lat]))
    sl[N-1-spec] = slice(*rng(c, n[spec]))
    # the data may have been squeezed: drop the length-1 axes (e.g. Stokes) it lacks
    sl_data = sl if data.ndim == N else [sl[k] for k in range(N)
                                         if n[N-1-k] > 1 or (N-1-k) in (lng, lat, spec)]
    sub = data[tuple(sl_data)]
    hdr = header.copy()
    new = wcs[tuple(sl)]
    if data.ndim != N:                                     # also make the WCS 3D
        new = new.sub([lng + 1, lat + 1, spec + 1])
        for k in list(hdr):
            if re.match(r"^(CTYPE|CRVAL|CDELT|CRPIX|CUNIT|CROTA|NAXIS|PC|CD)\d", k):
                del hdr[k]
    hdr.update(new.to_header())
    return sub, hdr

def velocity_axis(header, nchan):
    wcs = WCS(header)
    s = np.asarray(wcs.spectral.pixel_to_world_values(np.arange(nchan)), float).ravel()
    if wcs.wcs.ctype[wcs.wcs.spec].startswith("FREQ"):                  # Hz -> radio velocity, km/s
        f0 = header.get("RESTFRQ", F0)
        return C_KMS * (f0 - s) / f0
    return s / 1e3

def beam_pix(header):
    pix = proj_plane_pixel_scales(WCS(header).celestial)[0] * 3600
    return np.pi * header["BMAJ"] * header["BMIN"] * 3600**2 / (4 * np.log(2)) / pix**2

def spectrum_masked(data, header, mask, chunk_mb=256, label=""):
    """Sum of the masked voxels (mask > 0) in every channel, in Jy, and the number of masked voxels per channel."""
    nchan, ny, nx = data.shape
    if mask.shape != data.shape:
        raise ValueError(f"mask shape {mask.shape} differs from cube shape {data.shape}")
    bp = beam_pix(header)
    step = max(1, int(chunk_mb * 1e6 // (ny * nx * 4)))
    flux, nvox = [], []
    t0, nsteps = time.time(), math.ceil(nchan / step)
    for k, i in enumerate(range(0, nchan, step), 1):
        m = np.asarray(mask[i:i + step]) > 0
        if m.any():
            b = np.asarray(data[i:i + step], dtype=np.float32)
            flux.append(np.sum(b, axis=(1, 2), dtype=np.float64, where=m & np.isfinite(b)))
        else:                                                           # nothing masked: skip the cube
            flux.append(np.zeros(m.shape[0]))
        nvox.append(m.sum(axis=(1, 2)))
        if label and nsteps > 1:
            el = time.time() - t0
            print(f"\r  {label}: chunk {k}/{nsteps}, {el:.0f} s elapsed, ~{el / k * (nsteps - k):.0f} s left   ",
                  end="", flush=True)
    if label and nsteps > 1:
        print()
    vel = velocity_axis(header, nchan)
    o = np.argsort(vel)
    return dict(vel=vel[o], flux=np.concatenate(flux)[o] / bp, nvox=np.concatenate(nvox)[o])
 
def smooth_resample(vel, flux, grid, res):
    """Boxcar-smooth to ~res km/s and resample on `grid`."""
    k = int(round(res / np.median(np.abs(np.diff(vel)))))
    if k > 1:
        k += (k % 2 == 0)
        flux = convolve(flux, Box1DKernel(k), boundary="extend")
    return np.interp(grid, vel, flux)
 
def pixel_bounds(wcs_cel, shape, ra, dec):
    """Inclusive pixel bounds (x0, x1, y0, y1) of the pixels overlapping the sky corners; None if outside."""
    x, y = wcs_cel.wcs_world2pix(ra, dec, 0)
    ny, nx = shape
    x0, x1 = int(np.floor(min(x) + 0.5)), int(np.ceil(max(x) - 0.5))
    y0, y1 = int(np.floor(min(y) + 0.5)), int(np.ceil(max(y) - 0.5))
    x0, x1, y0, y1 = max(x0, 0), min(x1, nx - 1), max(y0, 0), min(y1, ny - 1)
    return None if (x0 > x1 or y0 > y1) else (x0, x1, y0, y1)
 
def match_bounds(wcs_from, bounds_from, wcs_to, shape_to):
    """Pixel bounds (inclusive) in cube `to` of exactly the sky region covered by `bounds_from` in cube `from`."""
    xa, xb, ya, yb = bounds_from
    x = [xa - 0.5, xb + 0.5, xa - 0.5, xb + 0.5]
    y = [ya - 0.5, ya - 0.5, yb + 0.5, yb + 0.5]                        # pixel-edge corners
    ra, dec = wcs_from.pixel_to_world_values(x, y)
    xt, yt = wcs_to.world_to_pixel_values(ra, dec)
    ny, nx = shape_to
    x0, x1 = max(0, int(np.rint(min(xt) + 0.5))), min(nx, int(np.rint(max(xt) + 0.5)))
    y0, y1 = max(0, int(np.rint(min(yt) + 0.5))), min(ny, int(np.rint(max(yt) + 0.5)))
    return x0, x1 - 1, y0, y1 - 1
 
def report(tag, label, spec):
    dv = np.median(np.abs(np.diff(spec["vel"])))
    has = spec["nvox"] > 0
    if not has.any():
        print(f"  {tag} {label}: no masked voxels")
        return
    v = spec["vel"][has]
    print(f"  {tag} {label}: {spec['nvox'].sum()} masked voxels, mask {v.min():.0f}-{v.max():.0f} km/s "
          f"(width {v.max() - v.min():.0f}), peak {spec['flux'].max():.4g} Jy, "
          f"integrated {np.nansum(spec['flux']) * dv:.3f} Jy km/s")
 
def draw(ax, specs, labels, grid):
    for spec, label in zip(specs, labels):
        ax.plot(grid, smooth_resample(spec["vel"], spec["flux"], grid, COMMON_RES)*u.Unit('Jy').to('mJy'), label=label)
    ax.axhline(0, ls="dashed", color="gray", lw=0.8)

def main():
    args = get_args()
    cube_high = args.cube_high
    cube_low = args.cube_low
    mask_high = args.mask_high
    mask_low = args.mask_low
    gal_name = args.gal_name # with xxx instead of galaxy number
    galaxies = args.galaxies
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    data_h, header_h, wcs_h = load_cube(cube_high)
    data_l, header_l, wcs_l = load_cube(cube_low)
    mask_h, _, _ = load_cube(mask_high)
    mask_l, _, _ = load_cube(mask_low)

    print("High resolution"); cube_extrema(wcs_h, header_h)
    print("Low resolution");  cube_extrema(wcs_l, header_l)

    #------------ Total spectrum (same volume and masked)
    sub_h, sub_header_h = make_subcube(data_h, header_h, wcs_h, header_l, wcs_l)
    sub_l, sub_header_l = make_subcube(data_l, header_l, wcs_l, header_h, wcs_h)

    msk_l, _ = make_subcube(mask_l, header_l, wcs_l, header_h, wcs_h)
    msk_h, _ = make_subcube(mask_h, header_h, wcs_h, header_l, wcs_l)

    labels = ["High resolution", "Low resolution"]
    print(f"common volume: {labels[0]} {sub_h.shape}, {labels[1]} {sub_l.shape}")
    total = [spectrum_masked(sub_h, sub_header_h, msk_h, label=labels[0]),
             spectrum_masked(sub_l, sub_header_l, msk_l, label=labels[1])]
    for l, s in zip(labels, total):
        report("total", l, s)
 
    grid = np.arange(max(s["vel"].min() for s in total), min(s["vel"].max() for s in total), COMMON_RES)
    vlim = None                                                # velocity range of the plots: mask extent + margin
    v = np.concatenate([s["vel"][s["nvox"] > 0] for s in total])
    if v.size:
        pad = 0.35 * (v.max() - v.min())
        vlim = (v.min() - pad, v.max() + pad)
 
    fig, ax = plt.subplots(figsize=(7, 6))                     # square figure
    draw(ax, total, labels, grid)
    ax.set_xlabel("Velocity (km/s)", fontsize=16); ax.set_ylabel("Flux (mJy)", fontsize=16); ax.legend(fontsize=12)
    if vlim: ax.set_xlim(vlim)
    ax.set_ylim(-7,180)
    ax.minorticks_on()
    ax.tick_params(which='major', direction='inout', length=8, labelsize=14)
    ax.tick_params(which='minor', direction='inout', length=5)
    ax.tick_params(axis='both', which='both', direction='inout', top=True, bottom=True, left=True, right=True)
    fig.tight_layout()
    out1 = os.path.join(out_dir, "total_spectrum.pdf")
    fig.savefig(out1, dpi=300)
    out1 = os.path.join(out_dir, "total_spectrum.png")
    fig.savefig(out1, dpi=300)
    print(f"saved {out1}")
 
    #------------ Spectra of the individual galaxies (same volume and masked)
    wcs_sub_h, wcs_sub_l = WCS(sub_header_h).celestial, WCS(sub_header_l).celestial
    results = {}
    for num in galaxies:
        ha = fits.getheader(gal_name.replace("XXX", str(num)))
        nx, ny = ha["NAXIS1"], ha["NAXIS2"]
        ra, dec = WCS(ha).celestial.pixel_to_world_values([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1])
        bl = pixel_bounds(wcs_sub_l, sub_l.shape[1:], ra, dec)  # the low-res cube decides the area
        if bl is None:
            print(f"NGC {num}: outside the common volume, skipped")
            continue
        x2a, x2b, y2a, y2b = bl
        x1a, x1b, y1a, y1b = match_bounds(wcs_sub_l, bl, wcs_sub_h, sub_h.shape[1:])
        specs = [spectrum_masked(sub_h[:, y1a:y1b + 1, x1a:x1b + 1], sub_header_h, msk_h[:, y1a:y1b + 1, x1a:x1b + 1]),
                 spectrum_masked(sub_l[:, y2a:y2b + 1, x2a:x2b + 1], sub_header_l, msk_l[:, y2a:y2b + 1, x2a:x2b + 1])]
        for l, s in zip(labels, specs):
            report(f"NGC {num}", l, s)
        results[num] = specs
 
    if results:
        ncol = 2
        nrow = max(3, math.ceil(len(results) / ncol))          # 3 rows x 2 columns
        fig, axs = plt.subplots(nrow, ncol, figsize=(10, 3.7 * nrow), squeeze=False)
        flat = axs.ravel()
        for ax, (num, specs) in zip(flat, results.items()):
            draw(ax, specs, labels, grid)
            ax.set_title(f"NGC {num}")
            if vlim: ax.set_xlim(vlim)
            ax.minorticks_on()
            ax.tick_params(which='major', direction='inout', length=8, labelsize=14)
            ax.tick_params(which='minor', direction='inout', length=5)
            ax.tick_params(axis='both', which='both', direction='inout', top=True, bottom=True, left=True, right=True)
        spare = flat[len(results):]
        for ax in spare:
            ax.axis("off")
        hnd, lab = flat[0].get_legend_handles_labels()
        if len(spare):
            spare[0].legend(hnd, lab, loc="center", fontsize=16, frameon=False)   # legend in the empty panel
        else:
            flat[0].legend(hnd, lab)
        for k, ax in enumerate(flat[:len(results)]):
            if k + ncol >= len(results):                       # nothing plotted below this panel
                ax.set_xlabel("Velocity (km/s)", fontsize=16)
        for ax in axs[:, 0]:
            ax.set_ylabel("Flux (mJy)", fontsize=16)
        fig.tight_layout()
        out2 = os.path.join(out_dir, "galaxies_spectra.png")
        fig.savefig(out2, dpi=300)
        out2 = os.path.join(out_dir, "galaxies_spectra.pdf")
        fig.savefig(out2, dpi=300)
        print(f"saved {out2}")
    

if __name__ == "__main__":
    main()