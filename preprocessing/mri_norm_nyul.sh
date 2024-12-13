#!/bin/bash
#SBATCH  --output=<PATH_TO_REPO>/sbatch_log/%j.out
#SBATCH  --error=<PATH_TO_REPO>/sbatch_log/log/%j.err  # where to store error messages
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
source <PATH_TO_REPO>/conda/shell/condabin shell.bash hook
source <PATH_TO_REPO>/conda/etc/profile.d/conda.sh

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

conda activate pytorch-CycleGAN-and-pix2pix

#with masks
nyul-normalize <PATH_TO_REPO>/normalization/before_single_patient/MR -m <PATH_TO_REPO>/normalization/before_single_patient/body_mask/ -o <PATH_TO_REPO>/normalization/Nyul_normalized -v --output-max-value 1 --output-min-value 0 --min-percentile 2 --max-percentile 98 -lsh <PATH_TO_REPO>/normalization/nyul_model_params.npy

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"
