

import nibabel as nib
import numpy as np
import os
import shutil
from glob import glob
from pydicom import read_file, dcmread
import pandas as pd
import math
from nibabel import processing
from metrics import structural_similarity_index, peak_signal_to_noise_ratio, mean_absolute_error, mean_squared_error
from develop_niftitodicom import convertNsave
import scipy

import pandas as pd



def merge_calc_dicom(created_slices,results_path,df_final_split, mask_dir,ct_input_dir,ct_max_value,ct_min_value,threshold_ct_air,threshold_ct_bones):
    merged_nifti_folder=os.path.join(results_path,"fake_nifti")
    df1 = created_slices
    df_all = df_final_split
    res_test = []
    if os.path.exists(merged_nifti_folder):
        print("!IMPORTANT! path={} exists. Removing it with all files inside".format(merged_nifti_folder))
        try:
            shutil.rmtree(merged_nifti_folder)
        except OSError as e:
            print("Error: %s : %s" % (merged_nifti_folder, e.strerror))
    os.makedirs(merged_nifti_folder)


    df1_grouped = df1.groupby(["Treatment_folder","N_real_slice"])

    # iterate over each group
    for group_name, df_group in df1_grouped:
        print('\ngroup {}('.format(group_name))

        ### TBD remove slices on the border

        slices_median=[]
        for row_index, row in df_group.iterrows():
            slice_path= row['Path']
            real_slice_index=int(group_name[1])
            treatment=group_name[0]

            nifti_file = nib.load(slice_path)
            nifti_array = nifti_file.get_fdata()
            slices_median.append(nifti_array)
            # print('\t{} {} {}  {}'.format(row_index, treatment,real_slice_index, slice_path))

        if real_slice_index not in [0, 41]:
            #save niftis
            slices_median = np.median(slices_median, axis=0)
            median_sl_nifti= nib.Nifti1Image(slices_median, np.eye(4))
            path_save= os.path.join(merged_nifti_folder,  treatment + "_" +str(real_slice_index)+".nii")
            nib.save(median_sl_nifti, path_save)

            fake_ct_numpy= slices_median
            # tbd, cover, if we do not have a mask file, all non background is an image pixels, bigger than -1024
            mask_path = glob(os.path.join(mask_dir, "mask_" + treatment + '*.nii'))
            mask_image = nib.load(mask_path[0])
            mask_nii_array = mask_image.get_fdata()
            slice_mask = mask_nii_array[:, :, int(real_slice_index)]

            real_slice_path = glob(os.path.join(ct_input_dir, treatment + '*.nii'))
            real_ct_image = nib.load(real_slice_path[0])
            real_ct_nii_array = real_ct_image.get_fdata()
            real_ct_nii_array[real_ct_nii_array < ct_min_value] = ct_min_value
            real_ct_nii_array[real_ct_nii_array > ct_max_value] = ct_max_value
            real_ct_numpy = real_ct_nii_array[:, :, int(real_slice_index)].astype(np.int16)

            mae= mean_absolute_error(real_ct_numpy, fake_ct_numpy, slice_mask)
            mse = mean_squared_error(real_ct_numpy, fake_ct_numpy, slice_mask)
            psnr = peak_signal_to_noise_ratio(real_ct_numpy, fake_ct_numpy, slice_mask)
            ssim = structural_similarity_index(real_ct_numpy, fake_ct_numpy)

            ##### get masked body, no air bubbles, for metrics calcs
            mask_no_air_real = np.zeros(real_ct_numpy.shape)
            mask_no_air_real[real_ct_numpy > threshold_ct_air] = 1
            mask_no_air_real[real_ct_numpy <= threshold_ct_air] = 0

            mask_no_air_fake = np.zeros(fake_ct_numpy.shape)
            mask_no_air_fake[fake_ct_numpy > threshold_ct_air] = 1
            mask_no_air_fake[fake_ct_numpy <= threshold_ct_air] = 0

            mask_air_joined = mask_no_air_real * mask_no_air_fake
            mae_no_air= mean_absolute_error(real_ct_numpy, fake_ct_numpy, mask_air_joined)

            ######get masked bone regions, for metrics calcs

            mask_bones = np.zeros(real_ct_numpy.shape)
            mask_bones[real_ct_numpy > threshold_ct_bones] = 1
            mask_bones[real_ct_numpy <= threshold_ct_bones] = 0

            mae_bones = mean_absolute_error(real_ct_numpy, fake_ct_numpy, mask_bones)

            res_test.append([mae, mse,  psnr, ssim, mae_no_air, mae_bones])

            #save fake dicom
            path_fake = os.path.join(results_path, "fake", treatment)
            os.makedirs(path_fake, exist_ok=True)
            slices_median = np.rot90(slices_median, -1)
            slices_median = np.fliplr(slices_median).astype(np.int16)
            convertNsave(slices_median, path_fake, df_all, treatment, real_slice_index)

    print("Results for test split, mean:")
    df = pd.DataFrame([
        pd.DataFrame(res_test, columns=['MAE', "MSE","PSNR", 'SSIM','MAE_NO_AIR','MAE_BONES' ]).mean().squeeze()
    ], index=[ 'Test set']).T

    print(df)

    print("Results for test split, standard deviation:")
    st_d_df = pd.DataFrame([
        pd.DataFrame(res_test, columns=['MAE', "MSE","PSNR", 'SSIM','MAE_NO_AIR','MAE_BONES' ]).std().squeeze()
    ], index=[ 'Test set']).T

    print(st_d_df)
    return df

