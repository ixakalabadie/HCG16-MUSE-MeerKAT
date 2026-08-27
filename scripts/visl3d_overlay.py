import os
import argparse
from visl3d import visual as vis
from visl3d import misc
from astropy.io import fits
import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.visualization import make_lupton_rgb
from astropy.coordinates import SkyCoord
import configparser
import json

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select output directory.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-a', '--halpha', dest='halpha_path', \
                        help='Path to Halpha datacube.')
    parser.add_argument('-r', '--radio', dest='hi_path', \
                        help='Path to HI datacube.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there if necessary.')
    args = parser.parse_args()
    return args

def get_isolevels(num):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        return json.loads(config_object.get(f"ngc{num}", "isolevels"))

# ── 1. Loading ───────────────────────────────────────────────────────────────

def get_mask(path):
    with fits.open(path) as hdul:
        mask = np.squeeze(hdul[0].data).astype(bool)
    return ~mask


def get_cube(path, mask_path):
    with fits.open(path) as hdul:
        data   = np.squeeze(hdul[0].data).astype(float)
        header = hdul[0].header
    wcs  = WCS(header, naxis=3)
    data = np.nan_to_num(data)
    if mask_path is not None:
        data *= get_mask(mask_path)
    return data, wcs, header   # data is (nZ, nDec, nRA) — standard FITS/numpy order


# ── 2. World-coordinate union bounding box ───────────────────────────────────

def _cube_world_bounds(data, wcs):
    """Min/max world coords over all 8 pixel corners."""
    nz, ny, nx = data.shape
    corners = np.array([[x, y, z]
                        for x in (0, nx-1)
                        for y in (0, ny-1)
                        for z in (0, nz-1)])
    world = wcs.all_pix2world(corners, 0)   # (8, 3): RA, Dec, vel
    return world.min(axis=0), world.max(axis=0)


def _union_bounds(data1, wcs1, data2, wcs2):
    lo1, hi1 = _cube_world_bounds(data1, wcs1)
    lo2, hi2 = _cube_world_bounds(data2, wcs2)
    return np.minimum(lo1, lo2), np.maximum(hi1, hi2)


# ── 3. Zero-pad one cube to cover the target world bounding box ───────────────

def _pad_to_bounds(data, wcs, header,
                   world_min, world_max):
    nz, ny, nx = data.shape

    pix_lo = wcs.all_world2pix([world_min], 0)[0]   # (x, y, z) pixel coords
    pix_hi = wcs.all_world2pix([world_max], 0)[0]
    pix_lo, pix_hi = np.minimum(pix_lo, pix_hi), np.maximum(pix_lo, pix_hi)

    pad_lo = np.maximum(0,  np.floor(-pix_lo)).astype(int)              # pixels to add before index 0
    pad_hi = np.maximum(0,  np.ceil(pix_hi - [nx-1, ny-1, nz-1])       # pixels to add after last index
                        ).astype(int)

    padded = np.pad(
        data,
        [(pad_lo[2], pad_hi[2]),   # Z   axis (numpy 0)
         (pad_lo[1], pad_hi[1]),   # Dec axis (numpy 1)
         (pad_lo[0], pad_hi[0])],  # RA  axis (numpy 2)
        mode='constant', constant_values=0,
    )

    new_header = header.copy()
    new_header['CRPIX1'] = header['CRPIX1'] + pad_lo[0]   # RA
    new_header['CRPIX2'] = header['CRPIX2'] + pad_lo[1]   # Dec
    new_header['CRPIX3'] = header['CRPIX3'] + pad_lo[2]   # Z
    new_header['NAXIS1'] = padded.shape[2]
    new_header['NAXIS2'] = padded.shape[1]
    new_header['NAXIS3'] = padded.shape[0]

    return padded, WCS(new_header, naxis=3), new_header


# ── 4. Coordinate arrays, units, ctypes, deltas ──────────────────────────────

def _axis_coords(wcs, shape_zyx):
    """
    1D world-coordinate arrays for each axis.
    Vary one pixel index at a time; appropriate for separable (orthogonal) WCS.
    Returns ra, dec, vel arrays in the native pixel-increasing direction.
    """
    nz, ny, nx = shape_zyx

    pix_ra  = np.zeros((nx, 3)); pix_ra[:,  0] = np.arange(nx)
    pix_dec = np.zeros((ny, 3)); pix_dec[:, 1] = np.arange(ny)
    pix_vel = np.zeros((nz, 3)); pix_vel[:, 2] = np.arange(nz)

    ra  = wcs.all_pix2world(pix_ra,  0)[:, 0]
    dec = wcs.all_pix2world(pix_dec, 0)[:, 1]
    vel = wcs.all_pix2world(pix_vel, 0)[:, 2]

    return ra, dec, vel


def _axis_info(header):
    """
    Units, CTYPE strings (magnitude/type), and absolute pixel scales
    for axes in (RA, Dec, Z) order (FITS axes 1, 2, 3).
    """
    units  = []
    ctypes = []
    deltas = []

    units.append(header.get('BUNIT', ''))  # e.g. 'Jy/beam' or 'erg/s/cm^2/AA'
    ctypes.append(header.get('BTYPE', ''))  # e.g. 'Intensity' or 'Flux'

    for i in [1, 2, 3]:
        units.append(header.get(f'CUNIT{i}', ''))
        ctypes.append(header.get(f'CTYPE{i}', f'AXIS{i}'))
        try:
            delta = abs(header[f'CD{i}_{i}'])
        except KeyError:
            delta = abs(header[f'CDELT{i}'])
        deltas.append(delta)

    return {
        'header': header,  # full FITS header for reference
        'units':  np.array(units),    # e.g. ['Jy/beam', 'deg',      'deg',     'm/s']
        'ctypes': np.array(ctypes),   # e.g. ['Intensity', 'RA---TAN', 'DEC--TAN','VRAD']
        'delta': np.array(deltas),   # absolute pixel scale, same units as above
    }


# ── 5. Main alignment function ───────────────────────────────────────────────

def _orient_cube(cube, wcs, shape_zyx):
    """
    Transpose (Z,Dec,RA) → (RA,Dec,Z) and flip each axis independently
    so that all world coordinates are strictly increasing.

    Returns the oriented cube and its (ra, dec, vel) coordinate arrays.
    """
    # ── Transpose (Z, Dec, RA) → (RA, Dec, Z) ──────────────────────────────
    cube = np.transpose(cube, (2, 1, 0))

    # ── Build coordinate arrays from cube1 (native pixel order) ────────────
    ra, dec, vel = _axis_coords(wcs, shape_zyx)   # in native pixel order

    # ── Flip each axis to ensure strictly increasing world coordinates ──────
    #    Both cubes and their coord array are flipped together.
    if ra[1] < ra[0]:
        cube = np.ascontiguousarray(np.flip(cube, axis=0))
        ra   = ra[::-1]

    if dec[1] < dec[0]:
        cube = np.ascontiguousarray(np.flip(cube, axis=1))
        dec  = dec[::-1]

    if vel[1] < vel[0]:
        cube = np.ascontiguousarray(np.flip(cube, axis=2))
        vel  = vel[::-1]

    return cube, ra, dec, vel

def align_cubes(path1, path2,
                mask_path1,
                mask_path2):
    """
    Align two FITS datacubes to a common world bounding box.

    Output axis layout
    ------------------
    Both cubes are returned as  (RA, Dec, Z)  with all axes in
    *increasing* coordinate order:
        cube[0]   → lowest RA
        cube[:,0] → lowest Dec
        cube[:,:,0] → lowest velocity / frequency

    Returns
    -------
    cube1, cube2 : ndarray, shape (nRA, nDec, nZ)

    coords : dict
        'ra'  : (nRA,)  world coords  [CUNIT1]
        'dec' : (nDec,) world coords  [CUNIT2]
        'vel' : (nZ,)   world coords  [CUNIT3]
        All in increasing order, matching cube axes.

    info1, info2 : dict  (one per cube; pixel scales differ between instruments)
        'units'  : (3,) str array  — physical units per axis, e.g. ['deg','deg','m/s']
        'ctypes' : (3,) str array  — FITS CTYPE strings, e.g. ['RA---TAN','DEC--TAN','VRAD']
        'deltas' : (3,) float array — absolute pixel scale in the axis units
    """
    data1, wcs1, hdr1 = get_cube(path1, mask_path1)
    data2, wcs2, hdr2 = get_cube(path2, mask_path2)

    world_min, world_max = _union_bounds(data1, wcs1, data2, wcs2)

    pad1, wcs1p, hdr1p = _pad_to_bounds(data1, wcs1, hdr1, world_min, world_max)
    pad2, wcs2p, hdr2p = _pad_to_bounds(data2, wcs2, hdr2, world_min, world_max)

    # ── Orient each cube independently ──────────────────────────────────────
    cube1, ra1, dec1, vel1 = _orient_cube(pad1, wcs1p, pad1.shape)
    cube2, ra2, dec2, vel2 = _orient_cube(pad2, wcs2p, pad2.shape)

    # coords from cube1; cube2's should match to within pixel-scale tolerance
    coords = {'ra': ra1, 'dec': dec1, 'vel': vel1}

    # ── Axis metadata (units/ctypes/deltas are instrument-specific) ─────────
    info1 = _axis_info(hdr1p)
    info2 = _axis_info(hdr2p)

    return cube1, cube2, coords, info1, info2

def transform2rms(l_cubes, units):
    rms_muse, rms_meerkat = misc.get_rms(l_cubes[0]), misc.get_rms(l_cubes[1])
    print(rms_muse, rms_meerkat)

    l_cubes[0] /= rms_muse
    l_cubes[1] /= rms_meerkat

    print('Max MUSE (rms) = ', np.max(l_cubes[0]), units[0])
    print('Max MeerKAT (rms) = ', np.max(l_cubes[1]), units[1])

    return l_cubes

def transform2coldens():
    pass

def get_decals(coords):
    config_object = configparser.ConfigParser()
    with open("hcg16_gals_params.ini","r") as file_object:
        config_object.read_file(file_object)
        decals_path = json.loads(config_object.get("general", "decals"))

    with fits.open(decals_path) as hdul:
        opt_data = hdul[0].data
        opt_header = hdul[0].header
        opt_wcs = WCS(opt_header).celestial

    p1 = SkyCoord(coords[0,0], coords[1,0], unit='deg')
    p1 = opt_wcs.celestial.world_to_pixel(p1)
    p2 = SkyCoord(coords[0,1], coords[1,1], unit='deg')
    p2 = opt_wcs.celestial.world_to_pixel(p2)

    opt_lims = np.array([np.sort([p1[0], p2[0]]), np.sort([p1[1], p2[1]])])
    opt_lims = np.round(opt_lims).astype(int)

    # don't know why dec is in the first axis
    r = opt_data[2][opt_lims[1,0]:opt_lims[1,1], opt_lims[0,0]:opt_lims[0,1]]
    g = opt_data[1][opt_lims[1,0]:opt_lims[1,1], opt_lims[0,0]:opt_lims[0,1]]
    b = opt_data[0][opt_lims[1,0]:opt_lims[1,1], opt_lims[0,0]:opt_lims[0,1]]
    # in header grz = [0,1,2]
    # z = 8931 AA, r = 6165 AA, g = 4686 AA
    opt_rgb = make_lupton_rgb(image_r=r, image_g=g, image_b=b, stretch=0.1, Q=10, minimum=0)

    return opt_rgb / 255

def create_visl3d_overlay(path1, path2, out_path, num):

    mask_path = f"{path1[:-5]}-mask.fits"

    muse_cube, meerkat_cube, coords, muse_info, meerkat_info = align_cubes(path1, path2, mask_path, None)
    coords = np.array([[coords['ra'][0], coords['ra'][-1]],
                      [coords['dec'][0], coords['dec'][-1]],
                      [coords['vel'][0], coords['vel'][-1]]])

    optical_data = get_decals(coords)
    l_cubes = [muse_cube, meerkat_cube]
    l_cubes = transform2rms(l_cubes, (muse_info['units'][0], meerkat_info['units'][0]))

    l_isolevels = get_isolevels(num)
    units = np.hstack(['RMS', muse_info['units'][1:]])
    mags = np.hstack(('Intensity', meerkat_info['ctypes'][1:]))
    delta = muse_info['delta'] # doesn't matter if it's muse or meerkat
    name = muse_info['header']['OBJECT'].split(' ')[0]

    images = [('DECaLS grz', misc.get_imcol(optical_data)[:2]), ('', None)] # ('blank', (None, None))
    sufix = ['_decals', ''] # '_blank'

    for i, image2d in enumerate(images):
        cube = vis.Cube(l_cubes, coords=coords, l_isolevels=l_isolevels, cmaps=['inferno_r', 'cividis_r'], units=units,
            mags=mags, name=name, delta=delta, resol=1, image2d=image2d)
        desc = misc.cube_info(cube)
        vis.createVis(cube, f'{out_path}/visl3d_ngc{num}{sufix[i]}', description=desc, pagetitle=name)
    
    print(f"Visualisations of NGC{num} successful.")

def create_visl3d_overlay_components(path1, path2, out_path, num, prefix=''):

    mask_path1 = f"{path1[:-5]}_component-mask.fits" # ones are galaxy but get_mask inverts it
    mask_path2 = f"{path1[:-5]}_component-mask-inverted.fits" # ones are galaxy but get_mask inverts it
    all_mask = f"{path1[:-5]}-mask.fits"

    muse_cube1, meerkat_cube, coords, muse_info, meerkat_info = align_cubes(path1, path2, mask_path1, None)
    muse_cube2, _, _, _, _ = align_cubes(path1, path2, mask_path2, None)
    
    muse_masked, _, _, _, _ = align_cubes(path1, path2, all_mask, None)
    muse_masked[muse_masked > 0] = 1
    muse_masked[muse_masked <= 0] = 0
    muse_masked = muse_masked.astype(bool)
    muse_masked = ~muse_masked

    muse_cube1 = muse_cube1*muse_masked
    muse_cube2 = muse_cube2*muse_masked

    coords = np.array([[coords['ra'][0], coords['ra'][-1]],
                      [coords['dec'][0], coords['dec'][-1]],
                      [coords['vel'][0], coords['vel'][-1]]])

    optical_data = get_decals(coords)
    l_cubes = [muse_cube1, muse_cube2, meerkat_cube]
    rms_muse, rms_meerkat = misc.get_rms(muse_cube1), misc.get_rms(meerkat_cube)
    l_cubes[0] /= rms_muse
    l_cubes[1] /= rms_muse
    l_cubes[2] /= rms_meerkat

    print('MUSE rms = ', rms_muse, muse_info['units'][0])
    print('MeerKAT rms = ', rms_meerkat, meerkat_info['units'][0])

    l_isolevels = get_isolevels(num)
    l_isolevels.append(l_isolevels[1])
    l_isolevels[1] = [1, 1.5, 2, 3, 4, 5, 10, 15, 20]
    units = np.hstack(['RMS', muse_info['units'][1:]])
    mags = np.hstack(('Intensity', meerkat_info['ctypes'][1:]))
    delta = muse_info['delta'] # doesn't matter if it's muse or meerkat
    name = muse_info['header']['OBJECT'].split(' ')[0]

    images = [('DECaLS grz', misc.get_imcol(optical_data)[:2]), ('', None)] # ('blank', (None,None))
    sufix = ['_decals', ''] # '_blank'

    for i, image2d in enumerate(images):
        cube = vis.Cube(l_cubes, coords=coords, l_isolevels=l_isolevels, cmaps=['inferno_r', 'Reds', 'cividis_r'], units=units, mags=mags, name=name, delta=delta, resol=1, image2d=image2d)
        desc = misc.cube_info(cube)
        vis.createVis(cube, f'{out_path}/{prefix}visl3d_ngc{num}{sufix[i]}', description=desc, pagetitle=name)
    
    print(f"Visualisations of NGC{num} with components successful.")

def main():
    args = get_args()
    halpha_path = args.halpha_path
    hi_path = args.hi_path
    out_dir = args.out_dir
    num = hi_path.lower().split('ngc')[-1][:3]
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    create_visl3d_overlay(halpha_path, hi_path, out_dir, num)

    if num == '835':
        halpha_path = halpha_path[:-5] + '_subcube.fits'

        create_visl3d_overlay_components(halpha_path, hi_path, out_dir, num, prefix='componets_')

if __name__ == "__main__":
    main()
