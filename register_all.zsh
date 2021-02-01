#! /usr/bin/env zsh

MNI_DIR="/home/fernando/episurg/mri/mni_icbm152_nl_VI_nifti"
k=120

dataset=$1

output_dir="${dataset}/mni"
mkdir -p ${output_dir}

for filepath in `ls ${dataset}/raw`
do
    filename=`basename ${filepath}`
    stem=${filename:t:r:r}
    echo ${stem}
    result="${output_dir}/${stem}_t1_post_on_mni.nii.gz"
    affine="${output_dir}/${stem}_t1_post_to_mni.txt"
    fmask="${dataset}/brain/${stem}.nii.gz"

    if [ ! -f ${fmask} ]; then
	echo $fg_bold[red] "${fmask} does not exist"
	continue
    fi

    if [ -f ${affine} ]; then
        echo "${affine} already exists"
        continue
    fi

    reg_aladin \
      -ref "${MNI_DIR}/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii.gz" \
      -rmask "${MNI_DIR}/cerebrum.nii.gz" \
      -flo "${dataset}/raw/${filename}" \
      -fmask ${fmask} \
      -ln 5 \
      -res ${result} \
      -aff ${affine}

    reg_resample \
      -ref "${MNI_DIR}/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii.gz" \
      -flo "${dataset}/raw/${filename}" \
      -res ${result} \
      -trans ${affine} \
      -inter 4
done

python \
  $HOME/git/ijcars-2020-scripts/make_previews.py \
  ${dataset}/mni $k ${dataset}/previews_mni
