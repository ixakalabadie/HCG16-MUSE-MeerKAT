# HCG16-MUSE-MeerKAT

This repository contains the code used to obtain the results presented in the article "A MUSE and MeerKAT study of dynamics and star formation in HCG 16". The pipeline is designed to reproduce the whole analysis and recreate the figures displayed in the article. The code is organized in a modular way, allowing for easy adaptation to other datasets. It also includes more scripts for additional analysis, such as BPT diagrams or component fitting.

## How to use

1. Clone the repository
2. Download the data. Most of the data are too heavy to be included in the repository; therefore, they must be obtained through other means. The masks available in the folder of each galaxy in the output folder (output/ngcXXX) have been generated manually with [iDaVIE](https://idavie.readthedocs.io/en/latest/). Instructions for downloading the data are provided in the [data](data/README.md) folder.
3. Build the environment container image from the definition (.def) file with [Singularity](https://docs.sylabs.io/guides/3.5/user-guide/index.html) or [Apptainer](https://apptainer.org/). It is specially useful for the MeerKAT pipeline ([meerkat_pipeline.py](scripts/meerkat_pipeline.py)), which requires [SoFiA-2](https://gitlab.com/SoFiA-Admin/SoFiA-2) and [Miriad](https://www.atnf.csiro.au/computing/software/miriad/). Otherwise, the python packages needed to run the rest of the scripts are defined within [hcg16_pipeline.def](environment/hcg16_pipeline.def).<br>
    ```
    sudo singularity/apptainer build hcg16_pipeline.sif hcg16_pipeline.def
    ```
    if you are not root:
    ```
    sudo singularity/apptainer --fakeroot build hcg16_pipeline.sif hcg16_pipeline.def
    ```
4. Run the environment container:
    ```
    singularity shell --bind path/to/folder/HCG16-MUSE-MeerKAT/:/home/username/ environment/hcg16_pipeline.sif /bin/bash
    ```
5. Run the pipeline within the environment:
    ```
    cd scripts
    bash run_pipeline.sh
    ```
6. The pipeline might take a while, a several minutes or a few hours are not unexpected, especially when running the MeerKAT pipeline and when creating the 3D models with ViSL3D.
7. All the output data will be stored in the [ouput](output) folder, including the SoFiA output, FITS files, and images. The images used in the article will be copied to the [article](article/figures) folder.
8. The pipeline should be able to run other datasets with minor modifications and changing input parameters. These parameters are found in [run_pipeline.sh](scripts/run_pipeline.sh), [hcg16_gals_params.ini](scripts/hcg16_gals_params.ini), and [hcg16_sofia_vopt.par](scripts/hcg16_sofia_vopt.par).