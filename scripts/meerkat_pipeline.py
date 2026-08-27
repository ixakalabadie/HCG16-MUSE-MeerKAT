import sys
import os
import argparse
import subprocess
from shutil import which
from astropy.io import fits
import argparse
import configparser
import json

def get_args():
    """
    This function parses and returns arguments passed in the command line.
    """
    description = 'Select dataset'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--datacube', dest='cube_path', \
                        help='Data cube to process.')
    parser.add_argument('-b', '--primary_beam', dest='pb_path', \
                        help='Primary beam of the datacube.')
    parser.add_argument('-o', '--outdir', dest='out_dir', \
                        help='Outup directory. New folders will be created there.')
    parser.add_argument('-p', '--parfile', dest='parfile', \
                        help='Sofia-2 paramenters file.')
    parser.add_argument('-s', '--scfind_threshold', dest='scfind_threshold',
                        help='Define scfind.threshold parameter.', default=5)
    parser.add_argument('-f', '--reliability_minpix', dest='reliability_minpix',
                        help='Define reliability.minPixels parameter.', default=0)
    parser.add_argument('-t', '--reliability_threshold', dest='reliability_threshold',
                        help='Define reliability.threshold parameter.', default=0.9)
    parser.add_argument('-e', '--reliability_enable', dest='reliability_enable',
                        help='Define reliability.enable parameter.', default=False)
    # parser.add_argument('-w', '--weight_cube', dest='weight_cube',
    #                     help='Weight cube for SoFiA.', default=False)
    args = parser.parse_args()
    return args

def squared_cube(cube, out_path):
    with fits.open(cube) as hdul:
        data_cube = hdul[0].data
        hdr = hdul[0].header
    output_cube = cube[:-5] + '_squared.fits'
    # Square the data
    squared_data_cube = data_cube * data_cube
    fits.writeto(output_cube, squared_data_cube, hdr, overwrite=True)
    return output_cube

def miriad_velsw(cube, out_path):
    """
    Convert frequency axis to velocity axis using MIRIAD task velsw

    Parameters
    ----------
    cube: str
         Input data cube whose spectral axis in frequency needs to converted to velocity
    out_path: str
        Output directory path to save results to
    """
    cube_name = os.path.basename(cube)
    cube_out = out_path + "/" + cube_name[:-5] + "_vopt.fits"
    os.makedirs(out_path, exist_ok=True)
    os.system("chmod a+x ./freqtovel.sh")
    command = ['./freqtovel.sh',
                cube, cube[:-5], cube_out]
    subprocess.run(command, check=True)
    return cube_out

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None

def eliminate_time(cat):
    '''Eliminates timestamp from sofia catalog. Updates the file

    Parameters
    ----------
    cat: str
        Path to sofia catalog
    '''
    # Read in the file
    with open(cat, 'r') as infile :
        lines = infile.readlines()
    # Replace the target string
    for i, line in enumerate(lines):
        if line[:7] == '# Time:':
            lines[i] = '# Time:\n'
    # Write the file out again
    with open(cat, 'w') as infile:
        for i in lines:
            infile.write(i)

def make_subcubes(in_path, out_path):
    in_name = os.path.basename(in_path)
    os.makedirs(out_path, exist_ok=True)
    out_dir = out_path

    with fits.open(in_path) as hdul:
        data = hdul[0].data
        data = data.squeeze()
        header = hdul[0].header

    for num in ['833', '835', '838', '839', '848']:
        config_object = configparser.ConfigParser()
        with open("hcg16_gals_params.ini","r") as file_object:
            config_object.read_file(file_object)
            lim = json.loads(config_object.get('ngc'+num, "meer_lims"))
        subcube = data[lim[2][0]:lim[2][1], lim[1][0]:lim[1][1], lim[0][0]:lim[0][1]]
        subcube_header = header.copy()
        subcube_header['NAXIS'] = 3
        subcube_header['NAXIS1'] = lim[0][1] - lim[0][0]
        subcube_header['NAXIS2'] = lim[1][1] - lim[1][0]
        subcube_header['NAXIS3'] = lim[2][1] - lim[2][0]
        try:
            subcube_header.remove('NAXIS4')
            subcube_header.remove('CRVAL4')
            subcube_header.remove('CDELT4')
            subcube_header.remove('CTYPE4')
            subcube_header.remove('CUNIT4')
        except:
            pass
        subcube_header['CRPIX1'] -= lim[0][0]
        subcube_header['CRPIX2'] -= lim[1][0]
        subcube_header['CRPIX3'] -= lim[2][0]

        hdu = fits.PrimaryHDU(subcube, subcube_header)
        hdul = fits.HDUList([hdu])
        os.makedirs(f"{out_dir}/ngc{num}", exist_ok=True)
        out_path = f"{out_dir}/ngc{num}/{in_name[:-5]}_ngc{num}.fits"
        hdul.writeto(out_path, overwrite=True)

def update_parfile(parfile, output_path, datacube,
              scfind_threshold, reliability_minpix,
              reliability_threshold, reliability_enable, weight_cube):
    '''Updates file with paramenters

    Parameters
    ----------
    parfile: str
        File contanining sofia parameters
    output_path: str
        Path of output folder
    datacube: str
        Path to datacube
    scfind_threshold: float
        Sofia parameter scfind.threshold
    reliability_threshold: float
        Sofia parameter reliability.threshold
    reliability_minpix: float
        Sofia parameter reliability.minPixels
    reliability_enable: str
        Sofia parameter reliability.enable
    weight_cube : str
        Sofia parameter input.weights
    Returns
    -------
    updated_parfile: str
        Path of file with updated parameters
    '''
    datacube_path = datacube
    datacube_name = os.path.basename(datacube)
    updated_parfile = os.path.join(output_path, datacube_name.split('_')[0] + '_sofia_updated_vopt.par')
    print(f'Updated parfile: {updated_parfile}')
    with open(parfile, 'r') as filein, open(updated_parfile, 'w') as fileout:
        lines = filein.read().replace('output_path', output_path)
        lines = lines.replace('datacube', datacube_path)
        lines = lines.replace('outname', f'{datacube_name}')
        lines = lines.replace('scfind_threshold', f'{scfind_threshold}')
        lines = lines.replace('reliability_minpix', f'{reliability_minpix}')
        lines = lines.replace('reliability_threshold', f'{reliability_threshold}')
        lines = lines.replace('reliability_enable', f'{reliability_enable}')
        if "pbcorr" in datacube:
            lines = lines.replace('weight_cube', weight_cube)
        fileout.write(lines)
    return updated_parfile

def execute_sofia(executable, output_catalog, updated_parfile):
    """
    Running sofia.

    Parameters
    ----------
    executable: str
        Name of executable, either sofia or sofia2,
        depends on how SoFia-2 environment was set up
    output_catalog: str
        Name of sofia output catalogue
    updated_parfile:
        updated sofia parameter file
    """
    print('Executing Sofia-2')
    subprocess.call([executable, updated_parfile])
    print("Catalogue ", output_catalog[:-4]+".xml")
    #subprocess.run(["sofia_image_pipeline", "-c", output_catalog[:-4]+".xml"])
    eliminate_time(output_catalog)

def run_sofia(parfile, output_path, cube_path,
              scfind_threshold, reliability_minpix,
              reliability_threshold, reliability_enable, weight_cube):
    """Run sofia

    Parameters
    ----------
    parfile: str
        File contanining parameters
    output_path: str
        Name of output folder
    cube_path: str
        Path to data cube
    scfind_threshold: float
        Sofia parameter scfind.threshold
    reliability_minpix: float
        Sofia parameter reliability.minPixels
    reliability_threshold: float
        Sofia parameter reliability.threshold
    reliability_enable: str
        Sofia parameter reliability.enable
    weight_cube: str
        Sofia parameter input.weights
    """
    cube_name = os.path.basename(cube_path.replace('.fits',''))
    print('Cube name: ', cube_name)
    os.makedirs(output_path, exist_ok=True)
    # join enters the / between the parts
    output_catalog = os.path.join(output_path, f'{cube_name}_cat.txt')
    print(f'Parfile: {parfile}')
    updated_parfile = update_parfile(parfile, output_path, cube_path,
            scfind_threshold, reliability_minpix,
            reliability_threshold, reliability_enable, weight_cube)
    if is_tool('sofia'):
        execute_sofia('sofia', output_catalog, f"{updated_parfile}")
    elif is_tool('sofia2'):
        execute_sofia('sofia2', output_catalog, f"{updated_parfile}")
    else:
        print('SoFia-2 not available. Please install it.')
        sys.exit(1)

def main():
    args = get_args()
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    cube_path = args.cube_path # '../data/hcg16_linehighres_masked.pbcorr.fits'
    pb_path = args.pb_path # '../data/hcg16_linehighres_masked.pb.fits'
    out_dir = args.out_dir # '../output'
    par_file = args.parfile # './hcg16_sofia_vopt.par'

    pb_vopt = miriad_velsw(pb_path, out_dir)

    cube_vopt = miriad_velsw(cube_path, out_dir)

    pb_squared = squared_cube(pb_vopt, out_dir)

    run_sofia(par_file, out_dir + '/sofia_output', cube_vopt, args.scfind_threshold,
        args.reliability_minpix, args.reliability_threshold, args.reliability_enable, pb_squared)
    # run_sofia(par_file, out_path + '/sofia_output', cube_vopt, 5.6, 0, 0.9, 'false', pb_squared)

    make_subcubes(cube_vopt, out_dir)

if __name__ == "__main__":
    main()
