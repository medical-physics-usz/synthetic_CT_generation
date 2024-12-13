#!/bin/bash



source  /home/<PATH_TO_REPO>/anaconda3/etc/profile.d/conda.sh
conda activate contrastive-unpaired-translation

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"

python -u <PATH_TO_REPO>/Preprocessing/CT_MR_preprocessing.py "$@"
python -u <PATH_TO_REPO>/Preprocessing/resampling.py "$@"
nyul-normalize <PATH_TO_REPO>/normalization/before_single_patient/MR -m <PATH_TO_REPO>/normalization/before_single_patient/body_mask/ -o <PATH_TO_REPO>/normalization/Nyul_normalized -v --output-max-value 1 --output-min-value 0 --min-percentile 2 --max-percentile 98 -lsh <PATH_TO_REPO>/normalization/nyul_model_params.npy
python -u <PATH_TO_REPO>/Preprocessing/slice_creator.py "$@"


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"