import os
import numpy as np
import nibabel as nib
global path_all
from glob import glob
import shutil


path_normalization_input= "<PATH_TO_DATA>/normalization/before_temp_air_overwrite/CT_reg"
path_normalization_CT_reg_norm= "<PATH_TO_DATA>/normalization/CT_reg_norm_temp_air_overwrite"

image_paths = glob(os.path.join(path_normalization_input,"*.nii") )

if os.path.exists(path_normalization_CT_reg_norm):
    print("!IMPORTANT! path={} exists. Removing it with all files inside".format(path_normalization_CT_reg_norm))
    try:
        shutil.rmtree(path_normalization_CT_reg_norm)
    except OSError as e:
        print("Error: %s : %s" % (path_normalization_CT_reg_norm, e.strerror))
os.makedirs(path_normalization_CT_reg_norm)

for image_path in image_paths:
    image=nib.load(image_path)
    patient = str(image_path.split('/')[9])
    # patient = str(image_path.split('/')[10])
    sample = np.array(image.get_fdata())
    print("mean before= {} patient {}.".format(np.mean(sample), patient))
    sample[sample < -1024] = -1024
    sample[sample > 1200] = 1200

    _min = -1024
    _max = 1200
    sample = (sample - _min)  / (_max - _min)
    print("mean after= {}.".format(np.mean(sample)))
    norm_nifti = nib.Nifti1Image(sample, image.affine)
    path_norm_nifti = os.path.join(path_normalization_CT_reg_norm,patient)
    nib.save(norm_nifti, path_norm_nifti)
