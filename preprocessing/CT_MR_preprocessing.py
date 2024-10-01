"""
This script processes DICOM 3D volumes by converting them into 3D NIFTI format and extracting required anatomical masks for sCT generation. The following logic flow is implemented:

1. **Input Handling**: Accepts command-line arguments to define the mode of operation (training [MR, CT Reg, CT preprocessing] or testing only for MR preprocessing) and the presence of RTSTRUCT files containing tumor  and tissue structure information.

2. **Directory Setup**: Configures paths for input data and output files, and associated Excel files.

3. **DICOM 3D to NIFTI 3D Conversion**: Converts DICOM files into NIFTI format, preserving essential imaging parameters [without resampling - it is done in next steps].

4. **Body Mask Extraction**: Depending on the availability of RTSTRUCT files, the script extracts body masks for CT and MR images. If no RTSTRUCT file is present, a thresholding-based workaround is employed to create a body mask.

5. **Tissue for Normalisation Mask Extraction**: Additional masks of tissues for n4+npeaks normalisation are extracted if there is RTSTRUCT file

6. **Overwritten CT_reg creation**: Overwritten in the area of air bubbles[air_or, tissue_or] 3D CT_reg nifti created to be used for final DVH computations if there is RTSTRUCT file

7. **Tumour Mask Extraction and ROI calculation**: Extraction of tumor masks  from RTSTRUCT files if available, to specify regions of interest (ROIs)  in axial direction around tumors, helping to mitigate the impact of corrupted  edge of FOV slices during training [edge of FOV will be taken out in resampling preprocessing step].

8. **DICOM params to excel**: Outputs relevant parameters and metadata into Excel files, including population characteristics and imaging details, which can be used for further analysis and deciding on which slices to use and which resampling params to apply.
"""

import os
import pandas as pd
import nibabel as nib
global path_all
from glob import glob
import argparse
from new_helpers import get_patient_info, setup_nifti_directories, calculate_n_cm_ring_from_shapeZ_center,\
    extract_dicom_parameters, convert_dicom_to_nifti, save_dicom_modality_info,CT_MR_preprocessing_cmd_argument_handling, \
    read_rs_file_and_extract_structure_names, get_and_save_body_mask_without_rs_file,get_masked_input,\
    get_and_save_body_mask, process_and_save_tissue_masks, process_fat_mask_extraction, \
    biggest_tumour_mask_extraction_and_ring_calculation, save_dicom_tags_to_excel, overwrite_and_save_ct_reg_overwritten_with_air_and_tissue

# ============================
# --> Constants Configuration
# ============================

path_root = "<YOUR PATH TO DATA>"
path_all = os.path.join(path_root + "data_exported")  # path containing all original CT dicom series + RT struct


#Path to data. Data structure for preprocessing: 1 folder per patient, eg Pat#_treatment_treatmentday Pat001_LIV_1a (1a - treatment planning (MR and CT reg, CT), 2a - treament day, only MR is taken),
# path_root = "/mnt/data/raoplan/sCT_data/second_paper/"
# path_all = os.path.join(path_root + "initial")  # path containing all original CT dicom series + RT struct

# path_all = "/srv/beegfs02/scratch/mr_to_ct/data/test_data_exported_single_patient/initial"
excel_folder = os.path.join(path_root , "excel")
# The Excel file will contain population and image parameters, as well as data about the available structure file and regions of interest (ROIs) to exclude corrupted files from training.
# further preprocessing and DICOM 3D volume reconstruction in postprocessing will be based on this data
path_excel = os.path.join(excel_folder + "/data_CT_MR_TEMP_second_paper.xlsx")
path_excel_bugged = os.path.join(excel_folder + "/bugged_list_shapes_TEMP_second_paper.xlsx")

# Threshold values for CT body mask creation, body masks are required for normalisation and focused ROI training
threshold_ct_body_mask = -400
# Threshold values for MR body mask creation, required if there is no rsstructure file and no ct_reg modality, to create a mr-based body mask
threshold_mr_body_mask = 20

#Radius (in mm) for calculating the region of interest (ROI) ring around the tumor in the axial direction.
#This value defines how much area to include around the tumor to avoid corrupted slices at the edge of FOV for training.
#If no RTSTRUCT file with tumor names is available, this value will determine the number of mm around the central axial slice
#to define the ROI.
ring_mm_calc = 50
#regex or list of tumour names to extract from RTSTRUCT file  to identify the ROI, in case rs structure file is present
tumour_names_to_extract = [
    "ptv2_v(1[0-9]|[1-9])_1[ab](_ph)?",
    "ptv1_(v(1[0-9]|[1-9])_(1[ab]|2a)|liver)"
]

#regex or list of tissue names to extract from RTSTRUCT file  to extract tissue masks for n4+npeaks normalisation and dosimetric air overwrite, in case rs structure file is present
masks_to_extract = {
    'liver': ['liver'],
    'air': ['air_or'],
    'bladder': ['water', 'bladder', 'gallbladder', 'water_or'],
    'tissue_or_water_or': ['water','tissue_or','tisse_or','tissues_or','tissueor', 'water_or', 'soft_tissue_or', 'soft_tissue_ph']
}
# names from masks_to_extract_to return the masks to create overwritten CT_reg, based on values below, for DVH computations
name_mask_tissue_or='tissue_or_water_or'
value_for_tissue_or_on_ct_reg = 7
name_mask_air_or ='air'
value_for_air_or_on_ct_reg = -1024

#parameters to extract fat tissue mask from CT reg for n4+npeaks normalisation
extract_fat_mask = True
threshold_fat_max = -60
threshold_fat_min = -160


# Info that has to be collected from the dicom file to excel in order to identify population and image parameters
dicom_tags = {'Rows', 'Columns', 'PixelSpacing', 'ImagePositionPatient', 'PatientAge', 'PatientBirthDate', 'StudyDate',
              'PatientSex', 'Modality', 'SeriesDescription',
              'BodyPartExamined'}
ignore_tags = {''}  # Info that is not needed from the dicom file
# ============================
# <--Constants Configuration
# ============================


def main(mode, is_rstruct_file_with_tumour):

    #argument handling
    is_rstruct_file_with_tumour, mode = CT_MR_preprocessing_cmd_argument_handling(mode, is_rstruct_file_with_tumour)
    modalities_to_preprocess_list = ["CT", "CT_reg", "MR"] if mode == "preprocessing_train" else ["MR"]

    if not os.path.exists(excel_folder):
        print("No folder for excel exists. Creating excel folder")
        os.makedirs(excel_folder)

    n_patients = 0
    df_bugged_list_of_shapes = pd.DataFrame()

    # for all patients transform all required modalities from DICOM to Nifti and create required tissue masks, detect ROI for training based on tumour size
    for path in glob(f'{path_all}/*/'):

        patient_folder, patient_number, treatment_site, treatment_day = get_patient_info(path)
        n_patients += 1

        for modality in modalities_to_preprocess_list:
            modality_dicom_path, path_nifti, modality_category = setup_nifti_directories(modality, path, patient_folder,
                                                                                         bool_remove_existing_nifti=True)
            # get all dicom files from the modality directory
            if os.path.exists(modality_dicom_path):
                #extract dicom parameters for mask manipulation
                sample_image, z_coords, slice_thickness, shape, pix_spacing, im_position = extract_dicom_parameters( modality_dicom_path, modality_category)

                #convert original dicom to nifti and save it
                nifti_filename = patient_folder + "_3D_input.nii"
                saved_nifti_file_path= convert_dicom_to_nifti(modality_dicom_path, path_nifti, nifti_filename)

                #get required original dicom parameters]
                if not os.listdir(path_nifti) == []:
                    nii_image = nib.load(saved_nifti_file_path)
                    nii_array = nii_image.get_fdata()

                    #saving initial modality dicom details
                    shapeZ_slice=nii_array.shape[2]
                    df_dicom_modality_info = save_dicom_modality_info(
                        shapeZ=shapeZ_slice,
                        slice_thickness=slice_thickness,
                        patient_folder=patient_folder,
                        treatment_day=treatment_day,
                        treatment_site=treatment_site,
                        modality=modality,
                        modality_dicom_path=modality_dicom_path,
                        path_nifti=path_nifti,
                        patient_number=patient_number
                    )

                    # Extract, process, and save the body mask and masked input (input_nifti * body_mask) based on the modality.
                    # If a structure file is available, all necessary masks will be extracted from it.
                    # If not, an alternative method for extracting body masks threshold based, primarily from CT_reg, then from MR if no CT_red, is  implemented.
                    if is_rstruct_file_with_tumour is False:
                        ct_reg_folder_path  = os.path.join(path, "CT_reg_nifti/")
                        body_mask =  get_and_save_body_mask_without_rs_file(nii_image, path_nifti, modality, threshold_ct_body_mask,
                                                               patient_folder, threshold_mr_body_mask,
                                                               ct_reg_folder_path)
                        if body_mask is None:
                            print(
                                "🔴 ERROR: NIfTI BODY mask creation failed for patient: '{}', modality: '{}'. NIfTI files were not created.".format(
                                    patient_folder, modality))
                            continue
                        # get which slices to save in the final file
                        five_cm_ring_min_z, five_cm_ring_max_z = calculate_n_cm_ring_from_shapeZ_center(shapeZ_slice, ring_mm_calc,
                                                                                        slice_thickness)

                        df_dicom_modality_info = save_dicom_modality_info(
                            shapeZ=shapeZ_slice,
                            slice_thickness=slice_thickness,
                            patient_folder=patient_folder,
                            treatment_day=treatment_day,
                            treatment_site=treatment_site,
                            modality=modality,
                            modality_dicom_path=modality_dicom_path,
                            path_nifti=path_nifti,
                            patient_number=patient_number,
                            five_cm_ring_min_z= five_cm_ring_min_z,
                            five_cm_ring_max_z= five_cm_ring_max_z,
                            n_slices_in_ring =five_cm_ring_max_z - five_cm_ring_min_z + 1
                        )

                    else:
                        ##### extract all masks and tumour center from RS file
                        rs_file_directory_path = os.path.join(path, "Plan/")
                        rs, names = read_rs_file_and_extract_structure_names(rs_file_directory_path)

                        body_mask = get_and_save_body_mask(
                            nii_image, path_nifti,  modality, threshold_ct_body_mask,
                            patient_folder, shape, z_coords, im_position, pix_spacing, df_bugged_list_of_shapes, rs, names
                        )

                        # If there is a bug in mask shapes, then previous function will add it to bugged shapes and continue with next modality
                        if body_mask is None:
                            print(
                                "🔴 ERROR: NIfTI BODY mask creation failed for patient: '{}', modality: '{}'. NIfTI files were not created.".format(
                                    patient_folder, modality))
                            continue

                        #####get all required tissue masks
                        if modality in ["CT_reg", "MR"]:

                            # Assuming rs, names, body_mask, shape, z_coords, im_position, pix_spacing, nii_image, and path_nifti are already defined
                            results, mask_tissue_or, mask_air_or = process_and_save_tissue_masks(
                                masks_to_extract, rs, names, body_mask, shape, z_coords, im_position, pix_spacing, nii_image,
                                path_nifti, name_mask_tissue_or, name_mask_air_or
                            )
                            print(results)

                            # create CT_reg with air and tissue overwritten
                            if modality == "CT_reg" and mask_tissue_or is not None and mask_air_or is not None:
                                masked_input_ct_overwritten= get_masked_input(nii_array, body_mask, modality)
                                overwrite_and_save_ct_reg_overwritten_with_air_and_tissue(mask_tissue_or, mask_air_or, masked_input_ct_overwritten,
                                    nii_image.affine, path_nifti, patient_folder,
                                    value_for_tissue_or_on_ct_reg, value_for_air_or_on_ct_reg, modality)

                            # extract fat tissue mask if required for further normalisation
                            if extract_fat_mask:
                                fat_mask_results = process_fat_mask_extraction( nii_image=nii_image,body_mask=body_mask, modality=modality,path_nifti=path_nifti,
                                    threshold_fat_min=threshold_fat_min,
                                    threshold_fat_max=threshold_fat_max
                                )
                                print(fat_mask_results)
                            else:
                                print("Fat mask extraction skipped.")

                            # biggest tumour mask extraction and 5 cm ring around tumour calculation
                            tumour_results = biggest_tumour_mask_extraction_and_ring_calculation(
                                rs=rs,
                                names=names,
                                nii_image=nii_image,
                                tumour_names_to_extract =tumour_names_to_extract,
                                shape=shape,
                                z_coords=z_coords,
                                im_position=im_position,
                                pix_spacing=pix_spacing,
                                path_nifti=path_nifti,
                                ring_mm_calc=ring_mm_calc,
                                slice_thickness=slice_thickness
                            )

                            # Check if extraction was successful
                            if tumour_results['status'] == 'success':
                                tumour_info = tumour_results['tumour_info']
                                mask_selected = tumour_results['mask_selected']
                                names_filtered_tumour_ptv1_v1 = tumour_results['ptv1_volumes']
                                names_filtered_tumour_ptv_ph = tumour_results['ptv2']

                                df_dicom_modality_info = save_dicom_modality_info(
                                    shapeZ=shapeZ_slice,
                                    slice_thickness=slice_thickness,
                                    patient_folder=patient_folder,
                                    treatment_day=treatment_day,
                                    treatment_site=treatment_site,
                                    modality=modality,
                                    modality_dicom_path=modality_dicom_path,
                                    path_nifti=path_nifti,
                                    patient_number=patient_number,
                                    tumour_min_z=tumour_info['tumour_min_z'],
                                    tumour_max_z=tumour_info['tumour_max_z'],
                                    tumour_center_z=tumour_info['tumour_center_z'],
                                    tumour_center_mass_z=tumour_info['tumour_center_mass_z'],
                                    tumour_size_in_slices_z=tumour_info['tumour_size_in_slices_z'],
                                    tumour_coord_z=tumour_info['tumour_coord_z'],
                                    five_cm_ring_min_z=tumour_info['five_cm_ring_min_z'],
                                    five_cm_ring_max_z=tumour_info['five_cm_ring_max_z'],
                                    n_slices_in_ring=tumour_info['n_slices_in_ring'],
                                    names=names,
                                    ptv1_volumes=names_filtered_tumour_ptv1_v1,
                                    ptv2=names_filtered_tumour_ptv_ph
                                )
                                print(f"Status: {tumour_results['status']}, "
                                      f"PTV1 Volumes: {tumour_results['ptv1_volumes']}, "
                                      f"PTV2 Volumes: {tumour_results['ptv2']}")

                            else:
                                print(f"🔴 ERROR: Issue occured extracting tumour masks: {tumour_results['error_message']}")

                    print("👍 NIfTI files successfully created for patient: '{}', modality: '{}'".format(patient_folder, modality))
                    #### dicom tags to excel
                    save_dicom_tags_to_excel(sample_image, dicom_tags, ignore_tags, df_dicom_modality_info, path_excel)

                else:
                    print("🔴 ERROR: NIfTI creation failed for patient: '{}', modality: '{}'. NIfTI files were not created.".format(patient_folder, modality))


    print("total n_patients or treatments created nifti= {}".format(n_patients))
    print("saving final excel")
    #### tbd_write_excel with timing
    df_bugged_list_of_shapes.to_excel(path_excel_bugged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Process DICOM 3D volumes: convert to NIFTI, extract required masks, "
            "and write Excel files with information on which slices to select to cover ROI."
        )
    )
    parser.add_argument(
        "-mode",
        choices=["preprocessing_train", "preprocessing_test"],
        required=True,
        help=(
            "Mode of operation: 'preprocessing_test' processes only MR images, "
            "'preprocessing_train' processes MR_reg, CT_reg, and CT when available."
        )
    )
    parser.add_argument(
        "-is_rstruct_file_with_tumour",
        required=True,
        help=(
            "Indicates presence of RTSTRUCT file with tumour (true/false). "
            "If set to 'false', NO tissue mask for n4+npeaks normalisation and air_or masks for DVH computation will be extracted, body masks will be extracted based on thresholds"
            "If set to 'true', all required tissue masks  will be extracted from RTSTRUCT file  and the ring in mm  will be calculated around the tumour in the Z axial direction. "
            "This is crucial for DVH computations and to choose a ROI in number of axial slices [recorded in excel] for further resampling to avoid corrupted slices at the edge of FOV. "
            "Kind reminder to check and adjust constants defined at the top of the file!"
        )
    )

    args = parser.parse_args()
    main(args.mode, args.is_rstruct_file_with_tumour)