# Data availability

The data to reproduce the results should be located in this folder. Below are instructions to download the required files. Note that the galaxies that we analyse are NGC 833 (HCG 16b), NGC 835 (HCG 16a), NGC 838 (HCG 16c), NGC 839 (HCG 16d), and NGC 848.

1. Emission line cubes from MUSE processed with PyPipe3D can be found here: [GAS_CUBES](https://ifs.astroscu.unam.mx/MUSE/GAS_CUBES/).
2. PyPipe3D flux_elines, SSP, and SFH files for MUSE observations can be found here: [dataproducts](https://ifs.astroscu.unam.mx/MUSE/dataproducts_NEW/)
3. MeerKAT data is not available online yet. Contact ixakalab@iaa.es to obtain the files.
4. DECaLS image of HCG 16 can be downloaded with<br>
    [https://www.legacysurvey.org/viewer/fits-cutout?ra=32.4750&dec=-10.2181&layer=ls-dr10-south&pixscale=0.524&width=3000&height=3000&bands=grz](https://www.legacysurvey.org/viewer/fits-cutout?ra=32.4750&dec=-10.2181&layer=ls-dr10-south&pixscale=0.524&width=3000&height=3000&bands=grz)

#### List of files

This is the list of files with their original names required to reproduce the analysis. XXX stands for the number of each NGC galaxy, that is, 833, 835, 838, 839, and 848. The name of the DECaLS image has been modified manually.

- GAS.NGCXX.cube.fits
- NGCXXX.SPP.cube.fits
- NGCXXX.SFH.cube.fits
- flux_elines.NGCXXX.cube.fits
- hcg16_line60_masked.pb_corr_vopt_mom0.fits
- hcg16_linehighres_masked.pb.fits
- hcg16_linehighres_masked.pbcorr.fits
- hcg16_full_3000_ps0.524_32.4750_-10.2181.fits (DECaLS image)
