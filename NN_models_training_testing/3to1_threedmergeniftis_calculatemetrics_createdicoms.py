
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



def merge_calc_dicom(nifti_3slices_dir,results_path,df_final_split, mask_dir,ct_input_dir,ct_max_value,ct_min_value,threshold_ct_air,threshold_ct_bones):


    merged_nifti_folder=os.path.join(results_path,"fake_nifti_3to1")
    df_all = df_final_split
    res_test = []
    if os.path.exists(merged_nifti_folder):
        print("!IMPORTANT! path={} exists. Removing it with all files inside".format(merged_nifti_folder))
        try:
            shutil.rmtree(merged_nifti_folder)
        except OSError as e:
            print("Error: %s : %s" % (merged_nifti_folder, e.strerror))
    os.makedirs(merged_nifti_folder)



    if os.path.exists(nifti_3slices_dir):
        image_paths = glob(os.path.join(nifti_3slices_dir, "*.nii"))
        for image_path in image_paths:

            #TBD slice naming snachala kakoi eto realno slice, potom ot kakogo grouping prishel

            # path_slice_fake=os.path.join(path_fake_nifti,treatment+"_"+ str(n_real_slice)+"_"+ slice+'.nii')

            file_name = str(image_path.split('/')[-1])
            treatment = "_".join(file_name.split('_')[:-2])


            print ("file_name")
            print(file_name)
            print ("treatment")
            print(treatment)


            real_slice_index = file_name.split('_')[-2]
            n_grouping = file_name.split('_')[-1].split('.')[0]
            print ("real_slice_index")
            print(real_slice_index)
            print ("n_grouping")
            print(n_grouping)


            if os.path.exists(image_path) and real_slice_index==n_grouping:


                nifti_file = nib.load(image_path)
                nifti_array = nifti_file.get_fdata()
            # print('\t{} {} {}  {}'.format(row_index, treatment,real_slice_index, slice_path))

                if real_slice_index not in [0, 41]:
                    #save niftis

                    centr_sl_nifti= nib.Nifti1Image(nifti_array, np.eye(4))
                    path_save= os.path.join(merged_nifti_folder,  treatment + "_" +str(real_slice_index)+".nii")
                    nib.save(centr_sl_nifti, path_save)

                    fake_ct_numpy= nifti_array
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
                    path_fake = os.path.join(results_path, "fake_3to1", treatment)
                    os.makedirs(path_fake, exist_ok=True)
                    nifti_array = np.rot90(nifti_array, -1)
                    nifti_array = np.fliplr(nifti_array).astype(np.int16)
                    convertNsave(nifti_array, path_fake, df_all, treatment, real_slice_index)
    print("mse")
    print(res_test)
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

    print("saving final excel")
    path_temp_x= "/srv/beegfs02/scratch/mr_to_ct/data/excel/MSE_test.xlsx"
    temp_mse= pd.DataFrame(res_test, columns=['MAE', "MSE","PSNR", 'SSIM','MAE_NO_AIR','MAE_BONES' ])
    temp_mse.to_excel(path_temp_x)

    return df





mask_dir="<PATH>/normalization/before/masks"
ct_input_dir="<PATH>/normalization/before/CT_reg"
path_excel_final_split = "<PATH>/excel/train_test_split_thesis_final.xls"
if os.path.exists(path_excel_final_split):
    df_all = pd.read_excel(path_excel_final_split, index_col=0)

threshold_ct_air=-400
threshold_ct_bones=250
ct_min_value=-1024
ct_max_value=1200



nifti_3slices_dir_lists=["<PATH>/data/latest_test/nifti_cyclegan_lsgan_pseudo3d_baseline_fix"]

for nifti_3slices_dir_list in nifti_3slices_dir_lists:
    nifti_3slices_dir = os.path.join(nifti_3slices_dir_list,"fake_nifti_3d")
    results_path=nifti_3slices_dir_list

    results=[]
    results = merge_calc_dicom(nifti_3slices_dir, results_path, df_all, mask_dir, ct_input_dir, ct_max_value,
                         ct_min_value, threshold_ct_air, threshold_ct_bones)
    print("<-The results above are for")
    print(nifti_3slices_dir)
