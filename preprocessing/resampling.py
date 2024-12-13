import os
import math
import numpy as np
import pandas as pd
import nibabel as nib
global path_all
from glob import glob
import shutil
from shutil import copyfile
from  new_helpers import  crop_or_pad,  affine_no_rotation
from nibabel import processing


path_for_norm_CT = "<YOUR PATH TO DATA>/CT_reg"
path_for_norm_MR = "<YOUR PATH TO DATA>/MR"
#body mask
path_new_masks = "<YOUR PATH TO DATA>/normalization/before_temp_air_overwrite/masks"
path_new_masks_air = "<YOUR PATH TO DATA>/normalization/before_temp_air_overwrite/masks_air"
path_new_masks_fat = "<YOUR PATH TO DATA>/normalization/before_temp_air_overwrite/masks_fat"
path_new_masks_liver = "<YOUR PATH TO DATA>/normalization/before_temp_air_overwrite/masks_liver"
path_new_masks_water_bladder = "<YOUR PATH TO DATA>/normalization/before_temp_air_overwrite/masks_water_bladder"

path_excel = "<YOUR PATH TO DATA>/excel/data_CT_MR_TEMP_second_paper.xlsx"
path_excel_resampling = "<YOUR PATH TO DATA>/excel/resampling_TEMP_second_paper_air_OR.xlsx"
df_whole_list= pd.DataFrame()


accepted_modal_folders= ["CT_reg","MR"]
accepted_tr_days= ["day0"]
crop_x_desired=256
crop_y_desired=256
desired_voxel_size_x = 1.6304
desired_voxel_size_y = 1.6304
desired_voxel_size_z =3
background_mri=0
background_ct=-1024

air_OR_option=True

#TBD: write transformations for MR fat mask
# removing folders with files if they exist and create new empty folders
for path_r in [path_for_norm_CT, path_for_norm_MR,path_new_masks_air, path_new_masks, path_new_masks_fat,path_new_masks_liver, path_new_masks_water_bladder ]:
    if os.path.exists(path_r):
        print("!IMPORTANT! path={} exists. Removing it with all files inside".format(path_r))
        try:
            shutil.rmtree(path_r)
        except OSError as e:
            print("Error: %s : %s" % (path_r, e.strerror))
    os.makedirs(path_r)

if os.path.exists(path_excel):
    df_all = pd.read_excel(path_excel, index_col=0)

    for index, line in df_all.iterrows():

        if line.TreatmentDay in accepted_tr_days and line.ModalityFolder in accepted_modal_folders and line.PathNIFTI!="" :


            if not os.path.exists(line.PathNIFTI):
                print("no nifti for Patient = {} treatment {} nifti_paths = {}".format(line.Patient,line.Treatment, line.PathNIFTI))
            else:
                if line.Modality=="MR":
                    nifti_norm_folder=path_for_norm_MR
                    background=background_mri
                else:
                    nifti_norm_folder = path_for_norm_CT
                    background = background_ct

                if air_OR_option and line.Modality=="CT":
                    image_path = os.path.join(line.PathNIFTI, line.Folder + '_3D_CT_air_overwrite.nii')
                else:
                    image_path = os.path.join(line.PathNIFTI,line.Folder+ '_3D_body.nii')
                mask_path = os.path.join(line.PathNIFTI, '3D_mask_body.nii')

                mask_path_air = os.path.join(line.PathNIFTI, '3D_mask_air.nii')
                mask_path_fat = os.path.join(line.PathNIFTI, '3D_mask_fat.nii')
                mask_path_liver = os.path.join(line.PathNIFTI, '3D_mask_liver.nii')
                mask_path_water_bladder = os.path.join(line.PathNIFTI, '3D_mask_bladder.nii')

                if image_path is None or mask_path is None:
                    print("!!!such a path for resampling does not exist={} or mask's {}".format(image_path, mask_path))
                else:
                    print("working on Patient = {} modality = {} ".format(line.Patient, line.Modality))
                    print(image_path)
                    image=nib.load(image_path)
                    image = affine_no_rotation(image)
                    nii_array_resampled = np.array(image.get_fdata())
                    initial_affine = image.affine

                    #body_mask
                    mask=nib.load(mask_path)
                    mask = affine_no_rotation(mask)
                    mask_resampled = np.array(mask.get_fdata())

                    mask_fat_flag=0
                    mask_liver_flag=0
                    mask_bladder_flag=0
                    mask_air_flag = 0
                    #load other masks for intensity normalization
                    if os.path.exists(mask_path_fat):
                        mask_fat_flag=1
                        mask_fat = nib.load(mask_path_fat)
                        mask_fat = affine_no_rotation(mask_fat)
                        mask_resampled_fat = np.array(mask_fat.get_fdata())
                    if os.path.exists(mask_path_liver):
                        mask_liver_flag=1
                        mask_liver = nib.load(mask_path_liver)
                        mask_liver= affine_no_rotation(mask_liver)
                        mask_resampled_liver = np.array(mask_liver.get_fdata())
                    if os.path.exists(mask_path_water_bladder):
                        mask_bladder_flag=1
                        mask_bladder = nib.load(mask_path_water_bladder)
                        mask_bladder= affine_no_rotation(mask_bladder)
                        mask_resampled_bladder = np.array(mask_bladder.get_fdata())
                    if os.path.exists(mask_path_air):
                        mask_air_flag=1
                        mask_air = nib.load(mask_path_air)
                        mask_air= affine_no_rotation(mask_air)
                        mask_resampled_air = np.array(mask_air.get_fdata())


                    pixel_size = np.array(line.PixelSpacing.replace("(", "").replace(")", "").replace(",", "").split()).astype(np.float32)
                    current_dim_x= int(pixel_size[0] * 10000) / 10000
                    current_dim_y = int(pixel_size[1] * 10000) / 10000

                    tumour_center_on_z=line.TumourZcentrSlide
                    # if the tumour center is not known, take the middle slice as the center
                    if line.TumourZcentrSlide==0:
                        tumour_center_on_z= int(nii_array_resampled.shape[2]/2)
                        print("tumour_center_on_z before_temp_air_overwrite= {}".format(tumour_center_on_z))

                    if current_dim_x!=desired_voxel_size_x or current_dim_y !=desired_voxel_size_y or line.SliceThickness!=desired_voxel_size_z:

                        # resample images
                        shape_before = nii_array_resampled.shape
                        masked_input_im_resampled = processing.resample_to_output(image, voxel_sizes=[desired_voxel_size_x,desired_voxel_size_y,desired_voxel_size_z], order=1, mode='constant', cval=background)
                        nii_array_resampled = masked_input_im_resampled.get_fdata()
                        initial_affine = image.affine

                        print(nii_array_resampled.shape)

                        #all masks should be the same for CT_reg and MR + CTreg has fat mask

                        if line.Modality=="CT":

                            #resample body mask
                            mask_nifti_resampled= processing.resample_to_output(mask, voxel_sizes=[desired_voxel_size_x,desired_voxel_size_y,desired_voxel_size_z], order=1, mode='constant', cval=0)
                            mask_resampled = mask_nifti_resampled.get_fdata()
                            initial_affine=mask_nifti_resampled.affine

                            #resample other masks if exist
                            if mask_fat_flag ==1:
                                mask_nifti_resampled_fat= processing.resample_to_output(mask_fat, voxel_sizes=[desired_voxel_size_x,desired_voxel_size_y,desired_voxel_size_z], order=1, mode='constant', cval=0)
                                mask_resampled_fat = mask_nifti_resampled_fat.get_fdata()
                            if mask_liver_flag ==1:
                                mask_nifti_resampled_liver= processing.resample_to_output(mask_liver, voxel_sizes=[desired_voxel_size_x,desired_voxel_size_y,desired_voxel_size_z], order=1, mode='constant', cval=0)
                                mask_resampled_liver = mask_nifti_resampled_liver.get_fdata()
                            if mask_bladder_flag ==1:
                                mask_nifti_resampled_bladder= processing.resample_to_output(mask_bladder, voxel_sizes=[desired_voxel_size_x,desired_voxel_size_y,desired_voxel_size_z], order=1, mode='constant', cval=0)
                                mask_resampled_bladder = mask_nifti_resampled_bladder.get_fdata()
                            if mask_air_flag ==1:
                                mask_nifti_resampled_air= processing.resample_to_output(mask_air, voxel_sizes=[desired_voxel_size_x,desired_voxel_size_y,desired_voxel_size_z], order=1, mode='constant', cval=0)
                                mask_resampled_air = mask_nifti_resampled_air.get_fdata()


                        #recalc tumour center slide
                        if line.SliceThickness!=desired_voxel_size_z:
                            tumour_center_on_z = int(tumour_center_on_z * line.SliceThickness/desired_voxel_size_z)
                            print("tumour_center_on_z after= {}".format(tumour_center_on_z))
                        print("resampling shape before= {}, new shape={}".format(shape_before, nii_array_resampled.shape))



                    #  crop or pad to the desired size
                    if nii_array_resampled.shape!=(crop_x_desired, crop_y_desired, nii_array_resampled.shape[2]):
                        shape_before = nii_array_resampled.shape
                        # for patients with displaced center of mass, adjust it
                        if line.Folder in ["PAT_LIST_HERE" ]:
                            nii_array_resampled=nii_array_resampled[30:,:,:]
                            mask_resampled=mask_resampled[30:,:,:]
                            if mask_fat_flag == 1:
                                mask_resampled_fat = mask_resampled_fat[30:,:,:]
                            if mask_liver_flag == 1:
                                mask_resampled_liver = mask_resampled_liver[30:,:,:]
                            if mask_bladder_flag == 1:
                                mask_resampled_bladder = mask_resampled_bladder[30:,:,:]
                            if mask_air_flag == 1:
                                mask_resampled_air = mask_resampled_air[30:,:,:]

                        elif line.Folder in ["PAT_LIST_HERE"]:
                            nii_array_resampled = nii_array_resampled[5:, :, :]
                            mask_resampled = mask_resampled[5:, :, :]
                            if mask_fat_flag == 1:
                                mask_resampled_fat = mask_resampled_fat[5:, :, :]
                            if mask_liver_flag == 1:
                                mask_resampled_liver = mask_resampled_liver[5:, :, :]
                            if mask_bladder_flag == 1:
                                mask_resampled_bladder = mask_resampled_bladder[5:, :, :]
                            if mask_air_flag == 1:
                                mask_resampled_air = mask_resampled_air[5:,:,:]

                        elif line.Folder in ["PAT_LIST_HERE"]:
                            nii_array_resampled = nii_array_resampled[15:, :, :]
                            mask_resampled = mask_resampled[15:, :, :]
                            if mask_fat_flag == 1:
                                mask_resampled_fat = mask_resampled_fat[15:, :, :]
                            if mask_liver_flag == 1:
                                mask_resampled_liver = mask_resampled_liver[15:, :, :]
                            if mask_bladder_flag == 1:
                                mask_resampled_bladder = mask_resampled_bladder[15:, :, :]
                            if mask_air_flag == 1:
                                mask_resampled_air = mask_resampled_air[15:,:,:]

                        elif line.Folder=="PAT_LIST_HERE":
                            nii_array_resampled = nii_array_resampled[20:,:-20,:]
                            mask_resampled = mask_resampled[20:,:-20,:]
                            if mask_fat_flag == 1:
                                mask_resampled_fat = mask_resampled_fat[20:,:-20,:]
                            if mask_liver_flag == 1:
                                mask_resampled_liver = mask_resampled_liver[20:,:-20,:]
                            if mask_bladder_flag == 1:
                                mask_resampled_bladder = mask_resampled_bladder[20:,:-20,:]
                            if mask_air_flag == 1:
                                mask_resampled_air = mask_resampled_air[20:,:-20,:]


                        elif line.Folder in ["PAT_LIST_HERE"]:
                            nii_array_resampled = nii_array_resampled[:-20,:,:]
                            mask_resampled = mask_resampled[:-20,:,:]
                            if mask_fat_flag == 1:
                                mask_resampled_fat = mask_resampled_fat[:-20,:,:]
                            if mask_liver_flag == 1:
                                mask_resampled_liver = mask_resampled_liver[:-20,:,:]
                            if mask_bladder_flag == 1:
                                mask_resampled_bladder = mask_resampled_bladder[:-20,:,:]
                            if mask_air_flag == 1:
                                mask_resampled_air = mask_resampled_air[:-20,:,:]

                        nii_array_resampled= crop_or_pad(nii_array_resampled, (crop_x_desired, crop_y_desired, nii_array_resampled.shape[2]),background)
                        # nii_array_resampled = nii_array_resampled.astype(np.int16)

                        if line.Modality == "CT":

                            #  crop or pad all masks, which exist
                            mask_resampled= crop_or_pad(mask_resampled, (crop_x_desired, crop_y_desired, mask_resampled.shape[2]),0)
                            # mask_resampled = mask_resampled.astype(np.int16)

                            print("affine before crop")
                            print(initial_affine, mask_resampled.shape)


                            if mask_fat_flag == 1:
                                mask_resampled_fat = crop_or_pad(mask_resampled_fat,
                                                             (crop_x_desired, crop_y_desired, mask_resampled_fat.shape[2]), 0)

                            if mask_liver_flag == 1:
                                mask_resampled_liver = crop_or_pad(mask_resampled_liver,
                                                             (crop_x_desired, crop_y_desired, mask_resampled_liver.shape[2]), 0)

                            if mask_bladder_flag == 1:
                                mask_resampled_bladder = crop_or_pad(mask_resampled_bladder,
                                                             (crop_x_desired, crop_y_desired, mask_resampled_bladder.shape[2]), 0)

                            if mask_air_flag == 1:
                                mask_resampled_air = crop_or_pad(mask_resampled_air,
                                                             (crop_x_desired, crop_y_desired, mask_resampled_air.shape[2]), 0)


                        print("crop or pad previous shape= {}, new shape={}".format(shape_before, nii_array_resampled.shape))

                    #save only 20 slices around tumour or around central Z slice if we do not know the tumour coordinates and do not take 10 first\last slices

                    first_slice_in_ring= tumour_center_on_z-20
                    last_slice_in_ring = tumour_center_on_z + 20
                    if first_slice_in_ring<10:
                        diff=10-first_slice_in_ring
                        first_slice_in_ring = first_slice_in_ring + diff
                        last_slice_in_ring = last_slice_in_ring + diff
                    if last_slice_in_ring>nii_array_resampled.shape[2]-10:
                        diff=10- (nii_array_resampled.shape[2]-last_slice_in_ring)
                        last_slice_in_ring = last_slice_in_ring - diff
                        first_slice_in_ring = first_slice_in_ring - diff
                    flag=0

                    # as CT_reg and MR  of listed patients have different Z shape and last slices are ok, first and last slices adjusted manually
                    if line.Folder in ["PAT_LIST_HERE"]:
                        first_slice_in_ring=46
                        last_slice_in_ring=86
                    elif line.Folder == "PAT_LIST_HERE":
                        first_slice_in_ring=38
                        last_slice_in_ring=78
                    elif line.Folder == "PAT_LIST_HERE":
                        first_slice_in_ring =30
                        last_slice_in_ring = 70
                    elif line.Folder == "PAT_LIST_HERE":
                        first_slice_in_ring =15
                        last_slice_in_ring = 55
                    elif line.Folder == "PAT_LIST_HERE":
                        first_slice_in_ring =19
                        last_slice_in_ring =59

                    # take an additional slice for pseudo 3D training (3 slices in axial direction as a NN input)
                    first_slice_in_ring = first_slice_in_ring-1
                    last_slice_in_ring=last_slice_in_ring+1

                    result = np.where(mask_resampled[:,:,first_slice_in_ring:last_slice_in_ring] == 1)
                    non_empty_z=np.unique(result[2])
                    written=0
                    if not np.array_equal(non_empty_z, range(mask_resampled[:,:,first_slice_in_ring:last_slice_in_ring].shape[2])):
                        print("NOT EQUAL -> there are empty slices")
                        flag=1
                    elif line.Modality=="CT":

                            #safe body mask (if there is no empty slices in the ring) + all other masks
                            path_mask_resized = os.path.join(path_new_masks, "mask_" + line.Folder + '_3D_body.nii')
                            mask_new = nib.Nifti1Image(mask_resampled[:,:,first_slice_in_ring:last_slice_in_ring], initial_affine)
                            nib.save(mask_new, path_mask_resized)

                            if mask_fat_flag == 1:
                                path_mask_resized_fat = os.path.join(path_new_masks_fat, "mask_" + line.Folder + '_3D_fat.nii')
                                mask_new_fat = nib.Nifti1Image(mask_resampled_fat[:,:,first_slice_in_ring:last_slice_in_ring],initial_affine)
                                nib.save(mask_new_fat, path_mask_resized_fat)
                            if mask_liver_flag == 1:
                                path_mask_resized_liver = os.path.join(path_new_masks_liver, "mask_" + line.Folder + '_3D_liver.nii')
                                mask_new_liver = nib.Nifti1Image(mask_resampled_liver[:,:,first_slice_in_ring:last_slice_in_ring], initial_affine)
                                nib.save(mask_new_liver, path_mask_resized_liver)
                            if mask_bladder_flag == 1:
                                path_mask_resized_bladder = os.path.join(path_new_masks_water_bladder, "mask_" + line.Folder + '_3D_bladder.nii')
                                mask_new_bladder = nib.Nifti1Image(mask_resampled_bladder[:,:,first_slice_in_ring:last_slice_in_ring], initial_affine)
                                nib.save(mask_new_bladder, path_mask_resized_bladder)
                            if mask_air_flag == 1:
                                path_mask_resized_air = os.path.join(path_new_masks_air, "mask_" + line.Folder + '_3D_air.nii')
                                mask_new_air = nib.Nifti1Image(mask_resampled_air[:,:,first_slice_in_ring:last_slice_in_ring], initial_affine)
                                nib.save(mask_new_air, path_mask_resized_air)

                    #if we have no empty slices, save the resized modality
                    if flag !=1:
                        path_masked_input_file_resized = os.path.join(nifti_norm_folder, line.Folder + '_3D_body.nii')
                        im = nib.Nifti1Image(nii_array_resampled[:,:,first_slice_in_ring:last_slice_in_ring], initial_affine)
                        nib.save(im, path_masked_input_file_resized)
                        written=1

                    n_ring=last_slice_in_ring-first_slice_in_ring
                    df_whole_list = df_whole_list.append({'Patient':line.Patient,'TreatmentDay':line.TreatmentDay,'Folder':line.Folder,'ModalityFolder':line.Modality,"first_slice_in_ring":first_slice_in_ring,"last_slice_in_ring":last_slice_in_ring,"n_ring":n_ring,"Shape":[nii_array_resampled.shape], "EmptySlices":flag, "Saved_nifti": written,"Fat_mask": mask_fat_flag,"Liver_mask": mask_liver_flag, "Bladder_mask": mask_bladder_flag }, ignore_index=True)

print("saving final excel")
df_whole_list.to_excel(path_excel_resampling)