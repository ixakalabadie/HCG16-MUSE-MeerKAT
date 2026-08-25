#!/bin/bash

galaxies=(835 833 838 839 848)

hi_cube="../data/hcg16_linehighres_masked.pbcorr.fits"
pb_cube="../data/hcg16_linehighres_masked.pb.fits"
sofia_par="./hcg16_sofia_vopt.par"
# Don't put numbers after NGC in gas_path and elines_path, just write XXX. It will be replaced by a number in the loop.
gas_path="../data/GAS.NGCXXX.cube.fits"
elines_path="../data/flux_elines.NGCXXX.cube.fits"
ssp_path="../data/NGCXXX.SSP.cube.fits"
sfh_path="../data/NGCXXX.SFH.cube.fits"
cat_path="sofia_output/hcg16_linehighres_masked.pbcorr_vopt_cat.txt"
out_dir="../output"
s2n=5

python meerkat_pipeline.py -d "$hi_cube" -b "$pb_cube" -o "$out_dir" -p "$sofia_par" -s 5.6 -f 0 -t 0.9 -e "false"

python hcg16_decals_HI60.py -o "$out_dir"

for num in "${galaxies[@]}"
do
    echo -e "\nRunning workflow for NGC${num}"

    python muse_pipeline.py -d "${gas_path/XXX/$num}" -e "${elines_path/XXX/$num}" -o "$out_dir/ngc${num}" -s "$s2n"

    python star_form_hist.py -p "${ssp_path/XXX/$num}" -f "${sfh_path/XXX/$num}" -o "$out_dir/ngc${num}"

    python radial_sfh.py -p "${ssp_path/XXX/$num}" -f "${sfh_path/XXX/$num}" -o "$out_dir/ngc${num}"

    python Halpha_SFR.py -a "${out_dir}/ngc${num}/NGC${num}_flux_Ha.fits" -b "${elines_path/XXX/$num}" -o "$out_dir/ngc${num}"

    # python bpt.py -e "${elines_path/XXX/$num}" -r "${out_dir}/ngcXXX/re_mass_sfh_ngcXXX.csv" -m "${out_dir}/ngc${num}/mask_re_ngc${num}.fits" -o "$out_dir/ngc${num}"

    python ha_flux_hi_mom1.py -i "$out_dir/ngc${num}/NGC${num}_flux_Ha.fits" -o "$out_dir/ngc${num}"

    python ha_flux_hi_coldens.py -i "$out_dir/ngc${num}/NGC${num}_flux_Ha.fits" -o "$out_dir/ngc${num}"

    python ha_mom1_hi_coldens.py -i "$out_dir/ngc${num}/NGC${num}_vopt_Ha.fits" -o "$out_dir/ngc${num}"

    python ha_mom1_hi_mom1.py -i "$out_dir/ngc${num}/NGC${num}_vopt_Ha.fits" -o "$out_dir/ngc${num}"

    # base=$(basename "$gas_path")
    # base="${base/XXX/$num}"
    # python fit_components.py -d "$out_dir/ngc${num}/${base/.fits/_Halpha_vopt.fits}" -o "$out_dir/ngc${num}"

    base=$(basename "$hi_cube")
    python visl3d_overlay.py -a "$out_dir/ngc${num}/GAS.NGC${num}.cube_Halpha_vopt.fits" -r "$out_dir/ngc${num}/${base/.fits/_vopt_ngc$num.fits}" -o "$out_dir/ngc${num}"
done

python obs_params.py -c ${out_dir}/${cat_path} -m "${gas_path/833/$num}" -o "${out_dir}"

python lindblad_ngc835.py -p "${ssp_path/XXX/835}" -o "${out_dir}/ngc835"

python compare_sfh.py -r "${out_dir}/ngcYYY/radial_XXX_ngcYYY.csv" -f "${out_dir}/ngcXXX/SFH_ngcXXX.csv" -e "${out_dir}/ngcXXX/re_mass_sfh_ngcXXX.csv" -g "${galaxies[@]}" -o "${out_dir}"

# Copy files to the article figures directory.
# add name after figures/ to set journal requirements
cp "${out_dir}/images/mass_accumulation_Re_norm.pdf" "../article/figures/"
cp "${out_dir}/images/hcg16_line60_decals.pdf" "../article/figures/"
cp "${out_dir}/images/comparison_radial_mass_all_convolved.pdf" "../article/figures/"
cp "${out_dir}/images/comparison_radial_sfh_all_specific_convolved.pdf" "../article/figures/"
cp "${out_dir}/images/comparison_radius_sfh_R1.0.pdf" "../article/figures/"
cp "${out_dir}/images/comparison_SFH_convolved.pdf" "../article/figures/"
cp "${out_dir}/images/convolved_SFH_HLR_0.25_specific.pdf" "../article/figures/"
cp "${out_dir}/ngc835/NGC835_omega_kappa_resonances.pdf" "../article/figures/"
cp "${out_dir}/ngc835/NGC835_continuum_resonances.pdf" "../article/figures/"

for num in "${galaxies[@]}"
do
    cp "${out_dir}/ngc${num}/ha_flux_hi_coldens_ngc${num}.pdf" "../article/figures/"
    cp "${out_dir}/ngc${num}/ha_mom1_hi_mom1_ngc${num}.pdf" "../article/figures/"
done

