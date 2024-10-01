
import sys
import os
import math
import pandas as pd
import pydicom
import cv2
import re
import string
import glob
from pydicom import dcmread
import numpy as np
import scipy
from scipy import ndimage
from skimage.draw import polygon
from skimage.measure import label
from skimage.segmentation import flood, flood_fill
from scipy.ndimage import morphology, measurements, filters
import nibabel as nib
from scipy.ndimage import morphology, measurements, filters, \
    binary_opening, binary_closing, binary_erosion, binary_dilation, binary_fill_holes
#import net
from skimage.measure import regionprops, label
import shutil
import dicom2nifti
import dicom2nifti.settings as settings
from pydicom import read_file, dcmread
import shutil
from nibabel import processing
import cv2
from scipy.ndimage import morphology, measurements, filters, \
    binary_opening, binary_closing, binary_erosion, binary_dilation, binary_fill_holes
settings.disable_validate_slice_increment()
settings.disable_validate_orthogonal()

def save_dicom_tags_to_excel(sample_image, dicom_tags, ignore_tags, df_dicom_modality_info, path_excel):
    """
    Extract DICOM tags and save them to an Excel file.

    Args:
        sample_image: The DICOM image used to extract tags.
        dicom_tags: Tags to search for in the DICOM image.
        ignore_tags: Tags to ignore during extraction.
        df_dicom_modality_info: DataFrame containing modality info to concatenate.
        path_excel: Path to the Excel file for saving the results.
    """
    # Extract DICOM tags
    dicom_dict_tags = search_dicom_tags(sample_image, dicom_tags, ignore_tags)
    df_dicom = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in dicom_dict_tags.items()})

    # Concatenate modality info with DICOM tags
    df_patient = pd.concat([df_dicom_modality_info.reset_index(drop=True),
                            df_dicom.reset_index(drop=True)], axis=1, join='outer', sort=True)

    # Check if Excel file already exists
    if os.path.exists(path_excel):
        df_all = pd.read_excel(path_excel)
        df_all = df_all[df_all.columns.drop(list(df_all.filter(regex='Unnamed.*')))]
        df_all = pd.concat([df_all.reset_index(drop=True), df_patient.reset_index(drop=True)], axis=0,
                           sort=True)
        df_all.to_excel(path_excel, index=False)  # Save updated DataFrame
    else:
        df_patient.to_excel(path_excel, index=False)  # Create new Excel file
def biggest_tumour_mask_extraction_and_ring_calculation(rs, names, nii_image, tumour_names_to_extract, shape, z_coords,
                                                        im_position, pix_spacing, path_nifti,
                                                        ring_mm_calc, slice_thickness):
    """
    Process tumour mask extraction, choose the biggest tumour volume, save it,
    and calculate the coordinates for a 5cm tumour ring.

    Args:
        rs (pydicom.Dataset): RTSTRUCT file containing contour data.
        names (list): List of structure names in the RTSTRUCT file.
        nii_image (Nifti1Image): Original NIfTI image object.
        tumour_names_to_extract (list): List of regex patterns for tumour extraction.
        shape (tuple): Shape of the image array.
        z_coords (array-like): Z coordinates of the image.
        im_position (tuple): Image position data.
        pix_spacing (tuple): Pixel spacing data.
        path_nifti (str): Path to save the extracted tumour mask.
        ring_mm_calc (float): Ring size in millimeters to calculate the 5 cm ring.
        slice_thickness (float): Thickness of each slice in the image.

    Returns:
        dict: A dictionary containing tumour mask extraction status and relevant information.
    """
    results = {}
    try:
        # Get the biggest tumour mask
        mask_selected, names_filtered_tumour_ptv1_v1, names_filtered_tumour_ptv_ph = biggest_tumour_mask_processing(
            rs, names, tumour_names_to_extract, shape, z_coords, im_position, pix_spacing, nii_image, path_nifti
        )

        # Calculate the 5cm tumour ring
        tumour_info = tumour_ring_calcs(mask_selected, ring_mm_calc, slice_thickness)

        # Return successful result along with tumour info
        results['status'] = 'success'
        results['tumour_info'] = tumour_info
        results['mask_selected'] = mask_selected
        results['ptv1_volumes'] = names_filtered_tumour_ptv1_v1
        results['ptv2'] = names_filtered_tumour_ptv_ph

    except Exception as e:
        # Handle errors and return them
        results['status'] = 'error'
        results['error_message'] = str(e)

    return results


def calculate_n_cm_ring_from_shapeZ_center(shapeZ, ring_mm_calc, slice_thickness):
    """
    Calculate the slice indices for a N cm tumour ring centered around the middle of the 3D volume.

    Args:
        shapeZ (int): Number of slices along the Z-axis (axial direction).
        ring_mm_calc (float): The desired ring size in millimeters (e.g., 50mm for 5 cm).
        slice_thickness (float): Thickness of each slice in millimeters.

    Returns:
        tuple: A tuple containing the min and max slice indices for the 5 cm tumour ring.
    """
    # Get the center slice index
    shapeZ_center_slice = (shapeZ - 1) // 2

    # Calculate the number of slices for the 5 cm tumour ring
    five_cm_ring_max_z = math.floor(shapeZ_center_slice + ring_mm_calc / slice_thickness)
    five_cm_ring_min_z = math.ceil(shapeZ_center_slice - ring_mm_calc / slice_thickness)

    # Ensure the ring slice indices are within the valid range
    five_cm_ring_max_z = min(five_cm_ring_max_z, shapeZ)
    five_cm_ring_min_z = max(five_cm_ring_min_z, 0)

    return five_cm_ring_min_z, five_cm_ring_max_z

def tumour_ring_calcs(mask_selected, ring_mm_calc, slice_thickness):
    """
    Calculate tumour coordinates and the 5cm tumour ring.

    Args:
        mask_selected (np.array): The selected tumour mask.
        ring_mm_calc (float): Ring size in millimeters to calculate the 5 cm ring.
        slice_thickness (float): Thickness of each slice in the image.

    Returns:
        dict: Tumour info including z coordinates and ring boundaries.
    """
    result = np.where(mask_selected == 1)

    if result[2].any():
        tumour_max_z = np.max(result[2])
        tumour_min_z = np.min(result[2])

        # Tumour size calculation
        tumour_coord_z = np.max(np.sum(mask_selected, axis=2))
        tumour_center_mass_z = np.round(measurements.center_of_mass(mask_selected))[2]

        # Calculate the 5 cm tumour ring
        five_cm_ring_max_z = math.floor(tumour_max_z + ring_mm_calc / slice_thickness)
        five_cm_ring_min_z = math.ceil(tumour_min_z - ring_mm_calc / slice_thickness)

        five_cm_ring_max_z = min(five_cm_ring_max_z, mask_selected.shape[2])
        five_cm_ring_min_z = max(five_cm_ring_min_z, 0)

        tumour_center_z = math.floor((tumour_max_z + tumour_min_z) / 2)

        # Tumour info to return
        tumour_info = {
            'tumour_min_z': tumour_min_z,
            'tumour_max_z': tumour_max_z,
            'tumour_center_z': tumour_center_z,
            'tumour_center_mass_z': tumour_center_mass_z,
            'tumour_size_in_slices_z': tumour_max_z - tumour_min_z + 1,
            'tumour_coord_z': tumour_coord_z,
            'five_cm_ring_min_z': five_cm_ring_min_z,
            'five_cm_ring_max_z': five_cm_ring_max_z,
            'n_slices_in_ring': five_cm_ring_max_z - five_cm_ring_min_z + 1
        }

        return tumour_info
    else:
        raise ValueError("No tumour coordinates found")

def biggest_tumour_mask_processing(rs, names, tumour_names_to_extract, shape, z_coords, im_position, pix_spacing,
                                   nii_image, path_nifti):
    """
    Extract tumour masks using regex patterns and select the largest tumour volume.

    Args:
        rs (pydicom.Dataset): RTSTRUCT file containing contour data.
        names (list): List of structure names in the RTSTRUCT file.
        tumour_names_to_extract (list): List of regex patterns for tumour extraction.
        shape (tuple): Shape of the image array.
        z_coords (array-like): Z coordinates of the image.
        im_position (tuple): Image position data.
        pix_spacing (tuple): Pixel spacing data.
        nii_image (Nifti1Image): NIfTI image object for affine information.
        path_nifti (str): Path to save the extracted tumour mask.

    Returns:
        tuple: Selected tumour mask, filtered tumour names for ptv1 and ptv2.
    """
    # Get tumour masks using regex patterns
    mask_ptv_ph, names_filtered_tumour_ptv_ph = get_tissue_mask_from_struct_file(
        rs, names, tumour_names_to_extract[0], shape, z_coords, im_position, pix_spacing, shape)

    mask_ptv1_v1, names_filtered_tumour_ptv1_v1 = get_tissue_mask_from_struct_file(
        rs, names, tumour_names_to_extract[1], shape, z_coords, im_position, pix_spacing, shape)

    # Choose the largest tumour volume
    if names_filtered_tumour_ptv1_v1 and names_filtered_tumour_ptv_ph:
        volume1_ptv1_v1 = np.count_nonzero(mask_ptv1_v1)
        volume2_ptv_ph = np.count_nonzero(mask_ptv_ph)

        if volume1_ptv1_v1 > volume2_ptv_ph:
            mask_selected = mask_ptv1_v1
        else:
            mask_selected = mask_ptv_ph
    elif names_filtered_tumour_ptv1_v1:
        mask_selected = mask_ptv1_v1
    elif names_filtered_tumour_ptv_ph:
        mask_selected = mask_ptv_ph
    else:
        raise ValueError("No matching PTV terms found")

    # Save selected tumour mask
    mask_sel_s = nib.Nifti1Image(mask_selected, nii_image.affine)
    path_mask_file = os.path.join(path_nifti, '3D_mask_tumour_ptv.nii')
    nib.save(mask_sel_s, path_mask_file)

    return mask_selected, names_filtered_tumour_ptv1_v1, names_filtered_tumour_ptv_ph

def process_fat_mask_extraction(nii_image, body_mask, modality, path_nifti, threshold_fat_min, threshold_fat_max):
    """
    Process fat mask extraction based on modality for the respective modality (CT or MR).
    Returns results with status and any errors encountered.

    Args:
        nii_image (Nifti1Image): Original NIfTI image object.
        body_mask (np.ndarray): Precomputed body mask to apply to extracted fat mask.
        modality (str): Imaging modality ('CT_reg' or 'MR').
        path_nifti (str): Path to save the extracted fat mask.
        threshold_fat_min (float): Minimum threshold for fat in CT scans.
        threshold_fat_max (float): Maximum threshold for fat in CT scans.

    Returns:
        dict: A dictionary with the status of the extraction process for each fat mask.
    """
    results = {}

    try:
        mask_shape = nii_image.shape
        nii_array = nii_image.get_fdata()  # Extract array from NIfTI image

        if modality == "CT_reg":
            try:
                mask_fat = np.zeros(mask_shape)
                mask_fat[nii_array <= threshold_fat_max] = 1
                mask_fat[nii_array < threshold_fat_min] = 0
                mask_fat = binary_opening(mask_fat, iterations=1).astype(np.int16)

                mask_im = nib.Nifti1Image(mask_fat * body_mask, nii_image.affine)
                path_mask_file = os.path.join(path_nifti, '3D_mask_fat.nii')
                nib.save(mask_im, path_mask_file)

                results['fat_CT'] = {'status': 'success'}
            except Exception as e:
                results['fat_CT'] = {'status': 'error', 'error_message': str(e)}

        elif modality == "MR":
            try:
                mask_fat_mr = np.zeros(mask_shape)
                fat_mr_threshold = np.mean(nii_array) * 1.7
                mask_fat_mr[nii_array >= fat_mr_threshold] = 1
                mask_fat_mr = binary_erosion(mask_fat_mr, iterations=1).astype(np.int16)
                mask_fat_mr = binary_opening(mask_fat_mr, iterations=1).astype(np.int16)

                mask_im = nib.Nifti1Image(mask_fat_mr * body_mask, nii_image.affine)
                path_mask_file = os.path.join(path_nifti, '3D_mask_fat_mr.nii')
                nib.save(mask_im, path_mask_file)

                results['fat_MR'] = {'status': 'success'}
            except Exception as e:
                results['fat_MR'] = {'status': 'error', 'error_message': str(e)}

    except Exception as e:
        results['overall_status'] = {'status': 'error', 'error_message': str(e)}

    return results


def overwrite_and_save_ct_reg_overwritten_with_air_and_tissue(mask_tissue_or, mask_air_or, masked_input_ct, affine, path_nifti, patient_folder,
                                     value_for_tissue_or_on_ct_reg, value_for_air_or_on_ct_reg, modality):
    """
    Overwrite CT reg image with tissue and air masks for further DVH computations, then save the result as a NIfTI file.

    Args:
        mask_tissue_or (ndarray): The tissue mask array to overwrite.
        mask_air_or (ndarray): The air mask array to overwrite.
        masked_input_ct (ndarray): The input CT image array to be overwritten.
        affine (Nifti1Image): Original affine of NIfTI image object.
        path_nifti (str): Path to the folder where NIfTI files will be saved.
        patient_folder (str): Name of the patient folder for file saving.
        value_for_tissue_or_on_ct_reg (int): The value to assign to tissue mask pixels in the CT image.
        value_for_air_or_on_ct_reg (int): The value to assign to air mask pixels in the CT image.
        modality (str): The imaging modality, the only acceptable is "CT_reg".

    Returns:
        None
    """
    if modality == "CT_reg" and mask_tissue_or is not None and mask_air_or is not None:
        # Overwrite the masked areas in the CT image
        masked_input_ct[mask_tissue_or == 1] = value_for_tissue_or_on_ct_reg
        masked_input_ct[mask_air_or == 1] = value_for_air_or_on_ct_reg
        masked_input_ct = masked_input_ct.astype(np.int16)

        # Create a new NIfTI image for the overwritten CT image
        masked_input_im = nib.Nifti1Image(masked_input_ct, affine)

        # Define the output file path and save the new NIfTI image
        path_masked_input_file = os.path.join(path_nifti, f'{patient_folder}_3D_body_overwritten_air_tissue.nii')
        nib.save(masked_input_im, path_masked_input_file)

    else:
        return None


def process_and_save_tissue_masks(masks_to_extract, rs, names, body_mask, shape, z_coords, im_position, pix_spacing,
                                  nii_image, path_nifti,  name_mask_tissue_or, name_mask_air_or):
    """
    Process and save the required tissue masks based on the list of masks to extract.

    Args:
        masks_to_extract (dict): Dictionary where keys are tissue names and values are lists of structure names in the RTSTRUCT file.
        rs (pydicom.Dataset): The RTSTRUCT DICOM file.
        names (list): List of structure names in the RTSTRUCT file.
        body_mask (ndarray): Body mask array for intensity normalization.
        shape (tuple): Shape of the output mask.
        z_coords (list): Z-coordinates for the mask generation.
        im_position (tuple): Image position used for mask creation.
        pix_spacing (tuple): Pixel spacing for the mask.
        nii_image (Nifti1Image): Original NIfTI image object.
        path_nifti (str): Path to the folder where NIfTI files will be saved.
        name_mask_tissue_or (str): Name of the tissue OR mask to extract for further DVH computations [CT reg overwrite].
        name_mask_air_or (str): Name of the air OR mask to extract for further DVH computations [CT reg overwrite].

    Returns:
        dict: Dictionary containing information about saved masks and any errors.
        ndarray: The extracted tissue OR mask.
        ndarray: The extracted air OR mask.
    """
    results = {}
    mask_tissue_or, mask_air_or = None, None

    for mask_name, structure_names in masks_to_extract.items():
        try:
            # Get the tissue mask for the current tissue
            mask, names_filtered = get_tissue_mask_from_struct_file(rs, names, structure_names, shape, z_coords,
                                                                    im_position, pix_spacing, shape)

            if names_filtered != []:
                # Post-process the mask with binary opening and multiply by the body mask
                mask = binary_opening(mask* body_mask, iterations=1).astype(np.int16)
                mask_im = nib.Nifti1Image(mask,  nii_image.affine)

                # Save the mask as a NIfTI file
                path_mask_file = os.path.join(path_nifti, f'3D_mask_{mask_name}.nii')
                nib.save(mask_im, path_mask_file)

                # Check if the current mask is the one for tissue OR or air OR to return for further CT reg overwrite
                if name_mask_tissue_or == mask_name:
                    mask_tissue_or = mask

                if name_mask_air_or == mask_name:
                    mask_air_or = mask

                # Update results
                results[mask_name] = {'status': 'success', 'filtered_names': names_filtered}
            else:
                results[mask_name] = {'status': 'no_names_found', 'filtered_names': []}

        except Exception as e:
            # Handle errors and update results
            results[mask_name] = {'status': 'error', 'error_message': str(e)}

    return results, mask_tissue_or, mask_air_or

def get_and_save_body_mask(nii_image, path_nifti, modality, threshold_ct_body_mask, patient_folder_name, shape,
                  z_coords, im_position, pix_spacing, df_bugged_list_shapes, rs, names):
    """
    Determine the body mask based on modality and save the results.

    Args:
        nii_image (nib.Nifti1Image): NIFTI image object.
        path_nifti (str): Directory path to save NIFTI files.
        path_RTSTRUCT (str): Path to the directory containing the RTSTRUCT file (for MR and CT_reg).
        modality (str): Modality name (e.g., "CT", "MR").
        threshold_ct_body_mask (float): Threshold for CT body mask.
        patient_folder_name (str): Folder name for the patient.
        shape (tuple): Shape of the image.
        z_coords (list): List of z-coordinates.
        im_position (list): Image position.
        pix_spacing (list): Pixel spacing.
        df_bugged_list_shapes (pd.DataFrame): DataFrame to append shape mismatches.
        rs (pydicom.dataset.FileDataset): The loaded RTSTRUCT DICOM file containing the structure set data, including the contours for each ROI. The rs file is required for extracting the contours corresponding to specific body parts or regions.
        names (list of str): A list of ROI names extracted from the rs file. Each name corresponds to a structure (e.g., "liver", "air_or") that is present in the RTSTRUCT DICOM file.

    Returns:
        np.ndarray: Masked input array or None if there is a shape mismatch.
    """
    if modality == "CT":
        return process_modality_body_mask_thresholding_only(nii_image, path_nifti, threshold_ct_body_mask, patient_folder_name, modality)
    elif modality in ["CT_reg", "MR"]:
        return process_mr_or_ct_reg_modality_body_mask(nii_image, path_nifti,  patient_folder_name, modality, shape,
                                             z_coords, im_position, pix_spacing, df_bugged_list_shapes, rs, names)
    else:
        raise ValueError(f"Unsupported modality to process body masks: {modality}")


def get_and_save_body_mask_without_rs_file(nii_image, path_nifti, modality, threshold_ct_body_mask, patient_folder_name, threshold_mr_body_mask, ct_reg_folder_path):
    """
    Extract and save a body mask based on the imaging modality, applying thresholding or using an existing mask.

    This function handles body mask extraction for both CT and MR modalities. For CT  and CT reg images, it applies a specified
    threshold to generate the mask. For MR images, it checks if a pre-existing body mask file ('3D_mask_body.nii') is
    available in the CT_reg folder. If the file is found, it copies and uses the mask for further processing. If no
    such mask is found, it applies thresholding based on the MR threshold value.

    Args:
        nii_image (nib.Nifti1Image): The NIFTI image object representing the original medical image.
        path_nifti (str): Directory path where the NIFTI files will be saved.
        modality (str): The imaging modality, such as 'CT' or 'MR'.
        threshold_ct_body_mask (float): Threshold value for extracting the body mask in CT images.
        patient_folder_name (str): The folder name for the current patient, used in constructing file paths.
        threshold_mr_body_mask (float): Threshold value for extracting the body mask in MR images, if no pre-existing mask is found.
        ct_reg_folder_path (str): Path to the directory containing the CT registration folder, where a pre-existing body mask may be stored.

    Returns:
        np.ndarray: The masked input array after applying the body mask, or None if there is an issue with processing.

    Raises:
        ValueError: If the modality is not supported.
    """
    if modality in ["CT", "CT_reg"]:
        return process_modality_body_mask_thresholding_only(nii_image, path_nifti, threshold_ct_body_mask, patient_folder_name, modality)
    elif modality in ["MR"]:
        # Check if 3D_mask_body.nii exists in ct_reg_folder_path then copy it and threshold based on it

        mask_body_file = os.path.join(ct_reg_folder_path, '3D_mask_body.nii')
        if os.path.exists(mask_body_file):
            # Copy the mask file to path_nifti
            dest_mask_file = os.path.join(path_nifti, '3D_mask_body.nii')
            shutil.copy(mask_body_file, dest_mask_file)
            # Load the copied mask file
            mask_image = nib.load(dest_mask_file)
            mask_array = mask_image.get_fdata()
            # Get the original NIfTI array of mr
            nii_array = nii_image.get_fdata()
            affine = nii_image.affine
            # Create mr masked input and save it
            masked_input = get_masked_input(nii_array, mask_array, modality)
            path_masked_input_file = os.path.join(path_nifti, patient_folder_name + '_3D_body.nii')
            save_nifti_image(masked_input, affine, path_masked_input_file)
            return mask_array
        else:
            return process_modality_body_mask_thresholding_only(nii_image, path_nifti, threshold_mr_body_mask, patient_folder_name, modality)
    else:
        raise ValueError(f"Unsupported modality to process body masks: {modality}")

def read_rs_file_and_extract_structure_names(rs_file_directory_path):
    """
    Extract the RTSTRUCT file and the names of structures from the DICOM RT file.

    Args:
        rs_file_directory_path (str): The full path to the directory containing the RTSTRUCT file.

    Returns:
        tuple: A tuple containing:
            - rs (pydicom.Dataset): The RTSTRUCT DICOM file.
            - names (list): A list of structure names from the RTSTRUCT file.
    """
    # Find the RTSTRUCT file
    rs_files = [file for file in os.listdir(rs_file_directory_path) if file.startswith('RTSTRUCT')]

    # Handle case where no RTSTRUCT file is found
    if not rs_files:
        raise FileNotFoundError(
            f"🔴 ERROR: No RTSTRUCT file found in the directory: {rs_file_directory_path}. "
            "Consider setting is_rstruct_file_with_tumour to False or provide a valid RTSTRUCT file."
        )
        sys.exit(1)

    # Read the first RTSTRUCT file found
    rs_file = rs_files[0]

    # Read the RTSTRUCT file
    rs = read_file(os.path.join(rs_file_directory_path, rs_file), force=True)

    # Extract names from the RTSTRUCT file
    try:
        names = [rs.StructureSetROISequence[j].ROIName for j in range(len(rs.ROIContourSequence))]
    except:
        print("🔴 ERROR in structure set")

    return rs, names
def process_mr_or_ct_reg_modality_body_mask(nii_image, path_nifti, patient_folder_name, modality, shape, z_coords,
                                  im_position, pix_spacing, df_bugged_list_of_shapes, rs, names):
    """
    Process MR or CT_reg modality to create and save body mask and masked input.

    Args:
        nii_image (nib.Nifti1Image): NIFTI image object.
        path_nifti (str): Directory path to save NIFTI files.
        rs
        names
        patient_folder_name (str): Folder name for the patient.
        modality (str): Modality name.
        shape (tuple): Shape of the image.
        z_coords (list): List of z-coordinates.
        im_position (list): Image position.
        pix_spacing (list): Pixel spacing.
        df_bugged_list_of_shapes (pd.DataFrame): DataFrame to append shape mismatches.
        rs (pydicom.dataset.FileDataset): The loaded RTSTRUCT DICOM file containing the structure set data, including the contours for each ROI. The rs file is required for extracting the contours corresponding to specific body parts or regions.
        names (list of str): A list of ROI names extracted from the rs file. Each name corresponds to a structure (e.g., "liver", "air_or") that is present in the RTSTRUCT DICOM file.


    Returns:
        np.ndarray: Mask or None if there is a shape mismatch.
    """

    # -------> this could be a function for all body masks, extract and save, based on the provided name and path
    # Get NIFTI array (data) and affine matrix from the NIFTI image object
    nii_array = nii_image.get_fdata()

    mask, names_filtered = get_tissue_mask_from_struct_file(rs, names, ['skin'], shape, z_coords, im_position,
                                                            pix_spacing, shape)

    # Process mask
    mask = process_and_smooth_mask(mask)

    path_mask_file = os.path.join(path_nifti, '3D_mask_body.nii')
    save_nifti_image(mask, nii_image.affine, path_mask_file)
    # <-------this could be a function for all body masks, extract and save


    #if different shapes - add to bugged shapes of lists and exit
    if nii_array.shape != mask.shape:
        df_bugged_list_of_shapes = df_bugged_list_of_shapes.append(
            {'Patient': patient_folder_name, 'Modality': modality, 'DicomShape': nii_array.shape,
             'MaskShape': mask.shape}, ignore_index=True)
        return None

    #save masked input
    masked_input = get_masked_input(nii_array, mask, modality)
    path_masked_input_file = os.path.join(path_nifti, patient_folder_name + '_3D_body.nii')
    save_nifti_image(masked_input, nii_image.affine, path_masked_input_file)

    return mask

def process_and_smooth_mask(mask):
    """
    Apply various processing steps to smooth the mask.

    Args:
        mask (np.ndarray): Original mask.

    Returns:
        np.ndarray: Processed and smoothed mask.
    """
    mask = binary_opening(mask, iterations=1).astype(np.int16)
    mask = binary_erosion(mask, iterations=3).astype(np.int16)
    mask = get_mask_biggest_contour(mask)
    mask = binary_dilation(mask, iterations=5).astype(np.int16)
    return binary_opening(mask, iterations=1).astype(np.int16)

def process_modality_body_mask_thresholding_only(nii_image, path_nifti, threshold_for_body_mask, patient_folder, modality):
    """
    Process CT modality to create and save body mask and masked input.

    Args:
        nii_image (nib.Nifti1Image): NIFTI image object.
        path_nifti (str): Directory path to save NIFTI files.
        threshold_for_body_mask (float): Threshold for CT body mask.
        patient_folder (str): Folder name for the patient.
        modality (str): Modality name ("CT", "CT_reg", or "MR").

    Returns:
        np.ndarray: Masked input array.
    """
    # Get NIFTI array (data) and affine matrix from the NIFTI image object
    nii_array = nii_image.get_fdata()
    affine = nii_image.affine

    # Generate body mask
    mask_threshold = get_body_mask_threshold(nii_array, threshold_for_body_mask)

    # Save the body mask as a NIFTI file
    path_mask_file = os.path.join(path_nifti, '3D_mask_body.nii')
    save_nifti_image(mask_threshold, affine, path_mask_file)

    # Create masked input and save it
    masked_input = get_masked_input(nii_array, mask_threshold, modality)
    path_masked_input_file = os.path.join(path_nifti, patient_folder + '_3D_body.nii')
    save_nifti_image(masked_input, affine, path_masked_input_file)

    return mask_threshold

def get_masked_input(nii_array, mask, modality):
    """
    Get the masked input array with the appropriate background based on the modality.

    Args:
        nii_array (np.ndarray): NIFTI array of the image.
        mask (np.ndarray): Binary mask array.
        modality (str): Modality name ("CT", "CT_reg", or "MR").

    Returns:
        np.ndarray: Masked input array with proper background values.
    """
    masked_input = nii_array * mask

    # Set the background value based on the modality
    if modality == "CT" or modality == "CT_reg":
        masked_input[mask == 0] = -1024  # Background for CT and CT_reg
    elif modality == "MR":
        masked_input[mask == 0] = 0  # Background for MR

    return masked_input.astype(np.int16)
def save_nifti_image(data, affine, file_path):
    """
    Save a NIFTI image to the specified path.

    Args:
        data (np.ndarray): Data to save.
        affine (np.ndarray): Affine matrix of the NIFTI image.
        file_path (str): Path where the NIFTI image will be saved.
    """
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, file_path)

def save_dicom_modality_info(patient_folder, treatment_day, treatment_site, modality, modality_dicom_path, path_nifti, shapeZ, slice_thickness, **kwargs):
    """
    Create a DataFrame with DICOM modality information for a given patient and modality.

    Args:
        shapeZ(str): number of slices
        slice_thickness (float): Thickness of the slices.
        patient_folder (str): Patient folder name.
        treatment_day (str): Treatment day.
        treatment_site (str): Treatment site.
        modality (str): Modality name (e.g., "CT", "MR").
        modality_dicom_path (str): Path to the DICOM files.
        path_nifti (str): Path to save the NIFTI files.
        **kwargs: Additional keyword arguments for more specific information.

    Returns:
        pd.DataFrame: DataFrame with the collected information.
    """
    sl_t = slice_thickness if slice_thickness else 0
    #### check whether pathDICOM and pathNIFTI is what we need
    df_data = {
        "ptv1_volumes": [kwargs.get('ptv1_volumes', 0)],
        "ptv2": [kwargs.get('ptv2', 0)],
        "Folder": patient_folder,
        'Patient': kwargs.get('patient_number', 0),
        'TreatmentDay': treatment_day,
        'Treatment': treatment_site,
        'ModalityFolder': modality,
        'PathDICOM': modality_dicom_path,
        'PathNIFTI': path_nifti,
        "TumourZfirstSlide": kwargs.get('tumour_min_z', 0),
        "TumourZlastSlide": kwargs.get('tumour_max_z', 0),
        "TumourZcentrSlide": kwargs.get('tumour_center_z', 0),
        "TumourZcenterOfmass": kwargs.get('tumour_center_mass_z', 0),
        "TumourSizeInSlicesZ": kwargs.get('TumourSizeInSlicesZ', 0),
        "TumourCoordinateZ": kwargs.get('tumour_coord_z', 0),
        "shapeZ": shapeZ,
        "5cmRingMinZ": kwargs.get('five_cm_ring_min_z', 0),
        "5cmRingMaxZ": kwargs.get('five_cm_ring_max_z', 0),
        "Nslices5cmRing": kwargs.get('N_slices_in_ring', 0),
        'SliceThickness': sl_t,
        "Names in RT": [kwargs.get('names', 0)]
    }

    df_paths = pd.DataFrame(df_data)
    return df_paths
def convert_dicom_to_nifti(modality_dicom_path, path_nifti, desired_filename):
    """
    Convert DICOM files to NIfTI format, save with a custom filename, and return the path of the saved file.

    Parameters:
    - modality_dicom_path (str): Path to the directory containing DICOM files.
    - path_nifti (str): Directory where the NIfTI file will be saved.
    - desired_filename (str): Desired name for the output NIfTI file, including the extension (e.g., 'output_file.nii').

    Returns:
    - str: The full path to the saved NIfTI file.

    Raises:
    - FileNotFoundError: If the DICOM path does not exist.
    - Exception: For errors during the conversion or renaming process.
    """
    if not os.path.exists(modality_dicom_path):
        raise FileNotFoundError(f"🔴 DICOM path {modality_dicom_path} does not exist.")

    # Convert DICOM to NIfTI
    if os.listdir(path_nifti):
        print("3D NIfTI already created.")
        nifti_file_path = os.path.join(path_nifti, desired_filename)
    else:
        try:
            dicom2nifti.convert_dir.convert_directory(
                modality_dicom_path,
                path_nifti,
                compression=False,  # No compression
                reorient=False  # No reorientation
            )

            # Define the default filename from the DICOM directory
            default_filename = os.listdir(path_nifti)[0]
            default_filepath = os.path.join(path_nifti, default_filename)

            # Define the new filename and path
            nifti_file_path = os.path.join(path_nifti, desired_filename)

            # Rename the file
            os.rename(default_filepath, nifti_file_path)
        except Exception as e:
            print(f"🔴  Error during DICOM to NIfTI conversion: {e}")
            raise

    return nifti_file_path

def extract_dicom_parameters(modality_dicom_path, modality_category):
    """
    Processes DICOM files in a specified directory, sorts them by z-coordinates, modifies metadata, and extracts image parameters.

    Args:
    - modality_dicom_path (str): Path to the directory containing DICOM files for the modality.
    - modality_category (str): Prefix/category to identify DICOM files within the directory.

    Returns:
    - sample_image (pydicom.Dataset): A sample DICOM image used to extract shape, pixel spacing, and image position.
    - z_coords (list): List of z-coordinates for the sorted slices.
    - slice_thickness (float): Calculated slice thickness between DICOM slices.
    - shape (tuple): Shape of the DICOM images (x, y, z).
    - pix_spacing (list): Pixel spacing in x and y dimensions.
    - im_position (list): Image position of the DICOM slices.
    """
    # Get all DICOM files that match the modality category
    dcm_files = [file for file in os.listdir(modality_dicom_path) if file.startswith(modality_category)]
    dcm_paths = [os.path.join(modality_dicom_path, file) for file in dcm_files if
                 os.path.isfile(os.path.join(modality_dicom_path, file))]

    # Sort DICOM files by z-coordinates
    sorted_dcm_paths, z_coords, error_sorting = sort_slices(dcm_paths)

    # Modify RepetitionTime and EchoTime metadata for each DICOM file
    for dcm_path in sorted_dcm_paths:
        ds = dcmread(dcm_path)
        ds.RepetitionTime = 0.
        ds.EchoTime = 0.
        ds.save_as(dcm_path)

    # Calculate slice thickness
    slice_thickness = [np.round(z_coords[j] - z_coords[i], 1) for i, j in
                       zip(np.arange(len(z_coords) - 1), np.arange(1, len(z_coords)))]
    slice_thickness = slice_thickness[0]  # Assume constant thickness across slices

    # Load a sample image to get shape and pixel spacing information
    sample_image = dcmread(sorted_dcm_paths[2])
    shape = (sample_image.Columns, sample_image.Rows, len(sorted_dcm_paths))
    pix_spacing = sample_image.PixelSpacing
    im_position = sample_image.ImagePositionPatient

    return sample_image, z_coords, slice_thickness, shape, pix_spacing, im_position


def setup_nifti_directories(modality, path, patient_folder, bool_remove_existing_nifti):
    if modality == "CT_reg":
        modality_dicom_path = os.path.join(path, modality)
        path_nifti = os.path.join(path, modality + "_nifti/")
        modality_category = "CT"
    else:
        modality_dicom_path = os.path.join(path, "Plan/" + modality)
        path_nifti = os.path.join(path, "Plan/" + modality + "_nifti/")
        modality_category = modality

    if os.path.exists(modality_dicom_path):
        if bool_remove_existing_nifti and os.path.exists(path_nifti):
            remove_existing_nifti(path_nifti, patient_folder, modality)
        os.makedirs(path_nifti)
    else:
        handle_missing_modality(patient_folder, modality)

    return modality_dicom_path, path_nifti, modality_category


def get_patient_info(path):
    patient_folder = str(path.split('/')[-2])
    patient_number = str(patient_folder.split('_')[0])
    treatment_site = "_".join(patient_folder.split('_')[1:-1])
    treatment_day = return_tr_day(str(patient_folder.split('_')[-1]))
    return patient_folder, patient_number, treatment_site, treatment_day

def handle_missing_modality(patient_folder, modality):
    print(
        "⚠️ IMPORTANT: The DICOM path does not exist for patient '{}' and modality '{}'. NIFTI files could not be created.".format(
            patient_folder, modality))

def remove_existing_nifti(path_nifti, patient_folder, modality):
    if os.path.exists(path_nifti):
        print(
            "⚠️ IMPORTANT: Existing NIFTI data found for patient '{}' and modality '{}'. Removing the directory and its contents.".format(
                patient_folder, modality))
        try:
            shutil.rmtree(path_nifti)
        except OSError as e:
            print("🔴 Error: Unable to remove NIFTI directory '{}'. Reason: {}".format(path_nifti, e.strerror))



def get_mask(contours=None, z=None, xy_pos=None, xy_res=None, shape_mask=None, value=1):
    pos_r = xy_pos[0]
    spacing_r = xy_res[0]
    pos_c = xy_pos[1]
    spacing_c = xy_res[1]
    label = np.zeros(shape_mask, dtype=np.int16)
    # print("Z {}".format(z))
    error = 0
    for c in contours:
        c_array = np.asarray(c)
        nodes = c_array.reshape((-1, 3))
        # print("Nodes {}".format(nodes))
        # assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        try:
            z_index = z.index(nodes[0, 2])
        except:
            try:
                c = [round(element, 2) for element in c]
                c_array = np.asarray(c)
                nodes = c_array.reshape((-1, 3))
                z = [round(element, 2) for element in z]
                z_index = z.index(nodes[0, 2])
            except:
                try:
                    c = [round(element, 1) for element in c]
                    c_array = np.asarray(c)
                    nodes = c_array.reshape((-1, 3))
                    z = [round(element, 1) for element in z]
                    z_index = z.index(nodes[0, 2])
                except:
                    error = 1
                    print(z)
                    print(c_array)
                    print("get_mask Error")
        # print(z_index)
        if not error:
            r = (nodes[:, 0] - pos_r) / spacing_r
            c = (nodes[:, 1] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = value
    return label

def get_sequences(path_patient, new_path_patient, files_dcm):
    if not os.path.exists(new_path_patient):
        os.makedirs(new_path_patient)
    names = []
    new_series = []
    count = 0
    for file in files_dcm:
        old_path = os.path.join(path_patient, file)
        print(old_path)
        ds = dcmread(old_path, force=True)

        try:
            name = str(ds[0x0008, 0x103E].value)
            print(name)
        except:
            try:
                name = str(ds['SeriesDescription'].value)
            except:
                print("Problem reading dicom in {}".format(path_patient))
                continue
        new_path_patient2 = os.path.join(new_path_patient, name)
        print(new_path_patient2)
        if name not in names:
            names.append(name)
        if not os.path.exists(new_path_patient2):
            os.makedirs(new_path_patient2)
        shutil.copy2(old_path, new_path_patient2)
    return names


def augment_crops(array, affine, new_dir):
    # crop_array_orig[crop_array_orig > 1800] = 0
    crop_array_orig = array.astype(np.int16)
    # crop_new = nib.Nifti1Image(crop_array_orig, affine)
    # nib.save(crop_new, os.path.join(new_dir, "crop_centered_0.9765_2.nii"))

    imageFlipper = sampling.imagetransformer.Flipper((0.5, 0., 0.))
    # Data augmentation: Rotations of up to +-10 degrees with a probability of 0.75
    imageRotator = net.sampling.imagetransformer.Rotator((0.75, 0.75, 0.), (15, 15, 0))
    # imageDeformer = sampling.imagetransformer.ElasticDeformer(1.,2,2)
    imageShifter = sampling.imagetransformer.RandomShifter((1., 1., 0.), (10, 10, 0))
    imageContrast = sampling.imagetransformer.ContrastChanger(0.9)
    transformers = [imageFlipper, imageRotator, imageShifter, imageContrast]

    for i in range(15):
        name = "crop_centered_0.9765_2_" + str(i) + ".nii"
        new_path = os.path.join(new_dir, name)
        if os.path.exists(new_path):
            os.remove(new_path)

    for i in range(15):
        crop_array = crop_array_orig
        for image_transformer in transformers:
            seed = i
            new_array = image_transformer.transform(crop_array, seed)
            new_array[new_array <= 0] = 0
            new_array = np.round(new_array)
            crop_array = new_array
        crop_array = crop_array.astype(np.int16)
        new_im = nib.Nifti1Image(crop_array, affine)
        name = "crop_centered_0.9765_2_" + str(i) + ".nii"
        new_path = os.path.join(new_dir, name)
        if os.path.exists(new_path):
            os.remove(new_path)
        nib.save(new_im, new_path)


def get_center_crop_CT(data):
    center_of_mass = np.round(measurements.center_of_mass(data))
    return center_of_mass


def remove_table(array):

    array = binary_opening(array, iterations=1).astype(np.uint8)
    # print("Opening")
    array = getLargestCC(array)
    # print("Largest")
    array = binary_dilation(array, iterations=10).astype(np.uint8)
    # print("Dilation")
    array = binary_fill_holes(array).astype(np.uint8)
    # print("Holes 1")
    array = binary_fill_holes(array, structure=np.ones((5, 5, 5))).astype(np.uint8)
    # print("Holes 5")
    # array = binary_fill_holes(array, structure= np.ones((10,10,10))).astype(np.uint8)
    # print("Holes 10")
    return array

def get_mask_biggest_contour(mask_ct):
    for i in range(mask_ct.shape[2]):
        inmask = np.expand_dims(mask_ct[:, :, i].astype(np.uint8), axis=2)
        ret, bin_img = cv2.threshold(inmask, 0.5, 1, cv2.THRESH_BINARY)
        (cnts, _) = cv2.findContours(np.expand_dims(bin_img, axis=2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # return None, if no contours detected
        if len(cnts) != 0:
            # based on contour area, get the maximum contour which is a body contour
            segmented = max(cnts, key=cv2.contourArea)
            bin_img[bin_img > 0] = 0
            a = cv2.drawContours(np.expand_dims(bin_img, axis=2), [segmented], 0, (255, 255, 255), -1)
            a[a > 0] = 1
            mask_ct[:, :, i] = a.squeeze()

    return mask_ct

def get_body_mask_threshold(nii_array,threshold_ct_body_mask):
    mask_ct = np.zeros(nii_array.shape)
    mask_ct[nii_array > threshold_ct_body_mask] = 1
    mask_ct[nii_array <= threshold_ct_body_mask] = 0
    mask_ct = binary_erosion(mask_ct, iterations=2).astype(np.uint8)
    mask_ct = get_mask_biggest_contour(mask_ct)
    mask_ct = binary_dilation(mask_ct, iterations=5).astype(np.int16)
    return mask_ct


def get_air_mask_ct_threshold(CT_body_array, mask_body,air_threshold, mask_limit):
    mask_ct = np.zeros(CT_body_array.shape)
    mask_ct[CT_body_array > air_threshold] = 0
    mask_ct[CT_body_array <= air_threshold] = 1
    mask_ct[mask_body == 0] = 0
    #mask_ct = binary_erosion(mask_ct, iterations=2).astype(np.uint8)
    #mask_ct = get_mask_biggest_contour(mask_ct)
    mask_ct = binary_dilation(mask_ct, iterations=5, mask= mask_limit).astype(np.int16)
    #mask_ct = binary_erosion(mask_ct, iterations=1).astype(np.uint8)
    return mask_ct

def get_limit_mask_ct_threshold(CT_body_array, bone_threshold):
    mask_body = np.zeros(CT_body_array.shape)
    mask_body[CT_body_array >= bone_threshold] = 1
    mask_body[CT_body_array < bone_threshold] = 0
    mask_limit_dilation = np.full(mask_body.shape, True)
    mask_limit_dilation[mask_body == 1] = False
    return mask_limit_dilation

def CT_MR_preprocessing_cmd_argument_handling(mode, is_rstruct_file_with_tumour):
    """
    Handle command-line arguments for the CT_MR preprocessing script.

    Args:
        mode (str): Mode of operation, must be 'preprocessing_train' or 'preprocessing_test'.
        is_rstruct_file_with_tumour (str): Indicates presence of RTSTRUCT file with tumour (true/false).

    Returns:
        tuple: A tuple containing the processed mode and is_rstruct_file_with_tumour value,
               and the list of modalities to preprocess.
    """
    # Argument handling
    if mode not in ["preprocessing_train", "preprocessing_test"]:
        print("Error: mode must be 'preprocessing_train' or 'preprocessing_test'")
        sys.exit(1)

    if is_rstruct_file_with_tumour.lower() in ['true', 't', 'yes', '1']:
        is_rstruct_file_with_tumour = True
    elif is_rstruct_file_with_tumour.lower() in ['false', 'f', 'no', '0']:
        is_rstruct_file_with_tumour = False
    else:
        print("Error: Argument -is_rstruct_file_with_tumour must be 'true' or 'false'.")
        sys.exit(1)

    return is_rstruct_file_with_tumour, mode

def get_tissue_mask_from_struct_file(rs, names_rs, names_to_extract, mask_shape, z_coords, im_position, pix_spacing,
                                     shape):
    """
    Extracts a tissue mask from a DICOM RTSTRUCT file based on specified tissue names.

    Args:
        rs (pydicom.dataset.FileDataset): The RTSTRUCT DICOM file dataset.
        names_rs (list of str): List of ROI names from the RTSTRUCT file.
        names_to_extract (list of str or str): List of specific tissue names or a regular expression pattern to match ROI names.
        mask_shape (tuple): The shape of the mask to be generated.
        z_coords (list of float): Z-coordinates of the image.
        im_position (list of float): Image position (origin) in the DICOM image.
        pix_spacing (list of float): Pixel spacing in the DICOM image.
        shape (tuple): Shape of the output mask.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The generated mask array.
            - list of str: Names of the tissues that were extracted.
    """
    names_filtered_tissues = []
    mask_tissues = np.zeros(mask_shape)

    for name in names_rs:
        if isinstance(names_to_extract, str):
            # Assume names_to_extract is a regular expression pattern
            pattern = re.compile(names_to_extract, re.IGNORECASE)
            names_filtered_tissues = [name for name in names_rs if pattern.match(name)]
        else:
            # Assume names_to_extract is a list of specific names
            if name.lower() in names_to_extract:
                names_filtered_tissues = names_filtered_tissues + [name]

    if names_filtered_tissues!= []:
        names_filtered_tissues.sort(reverse=True)

        for count, name in enumerate(names_filtered_tissues):
            count += 1
            for j in range(len(rs.ROIContourSequence)):
                if rs.StructureSetROISequence[j].ROIName == name:
                    index = j
                    # print("ROI name= {} ".format(rs.StructureSetROISequence[j].ROIName))
                    break

            contour_coords_tissue = []
            for s in rs.ROIContourSequence[index].ContourSequence:
                try:
                    contour_data = s[0x3006, 0x0050].value
                    values = [round(float(x), 1) for x in contour_data]
                    contour_coords_tissue.append(values)
                except:
                    pass

            mask_tissues += get_mask(contours=contour_coords_tissue, z=z_coords, xy_pos=im_position, xy_res=pix_spacing,
                                     shape_mask=shape, value=count)
            mask_tissues[mask_tissues != 0] = 1
        mask_tissues = mask_tissues.astype(np.int16)

    return mask_tissues, names_filtered_tissues

# def get_tissue_mask_from_struct_file(rs, names,names_to_extract, mask_shape,z_coords, im_position, pix_spacing,
#                                     shape):
#     names_filtered_tumour_ptv_ph = []
#     mask_ptv_ph = np.zeros(mask_shape)
#
#     for name in names:
#         if name.lower() in names_to_extract:
#             names_filtered_tumour_ptv_ph = names_filtered_tumour_ptv_ph + [name]
#     if names_filtered_tumour_ptv_ph != []:
#         names_filtered_tumour_ptv_ph.sort(reverse=True)
#
#         for count, name in enumerate(names_filtered_tumour_ptv_ph):
#             count += 1
#             for j in range(len(rs.ROIContourSequence)):
#                 if rs.StructureSetROISequence[j].ROIName == name:
#                     index = j
#                     print("ROI name= {} ".format(
#                         rs.StructureSetROISequence[j].ROIName))
#                     break
#
#             contour_coords_tumour = []
#             for s in rs.ROIContourSequence[index].ContourSequence:
#                 try:
#                     contour_data = s[0x3006, 0x0050].value
#                     values = [round(float(x), 1) for x in contour_data]
#                     contour_coords_tumour = contour_coords_tumour + [values]
#                 except:
#                     pass
#
#             mask_ptv_ph += get_mask(contours=contour_coords_tumour, z=z_coords, xy_pos=im_position, xy_res=pix_spacing,
#                                     shape_mask=shape, value=count)
#             mask_ptv_ph[mask_ptv_ph != 0] = 1
#         mask_ptv_ph = mask_ptv_ph.astype(np.int16)
#     return mask_ptv_ph,names_filtered_tumour_ptv_ph




def findFirstSlice(array):
    i_sum = 0
    # i_mean = 0
    slice = array.shape[2] - 1
    # print(np.sum(array[...,slice]))
    while round(i_sum) <= 4096:
        i_sum = np.sum(array[..., slice])
        # print("Sum {}".format(i_sum))
        # i_mean = np.mean(array[...,slice])
        slice = slice - 1
        if slice == 0:
            print("No slice found")
            break
    first_slice = (array.shape[2] - 1 - slice)
    return int(first_slice)


def findFirstSlice_MR(array):
    # i_sum = 0
    i_mean = 0
    slice = array.shape[2] - 1
    # print(np.sum(array[...,slice]))
    while round(i_mean) <= 4:
        # i_sum = np.sum(array[...,slice])
        i_mean = np.mean(array[..., slice])
        print(i_mean)
        slice = slice - 1
    first_slice = (array.shape[2] - 1 - slice)
    return first_slice


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def segment_out_brain(data, threshold):
    thresh_data = data
    thresh_data[data < threshold] = 0
    sobel_data = np.hypot(filters.sobel(thresh_data, axis=0), filters.sobel(thresh_data, axis=1))
    sobel_data = morphology.binary_dilation(sobel_data, iterations=5)
    sobel_data = morphology.binary_closing(sobel_data, iterations=4)
    mask_brain = sobel_data.astype(np.int16)
    center_of_mass = measurements.center_of_mass(mask_brain)
    return mask_brain, center_of_mass


def crop_HN(im, center, voxel_sizes, size_x_mm, size_y_mm, size_z_mm, first_slice):
    im_array = im.get_fdata(caching='unchanged')
    im_array = im_array.astype(np.float32)
    size_x, size_y, size_z = im.shape
    # print('Initial shape of image {}'.format(im_array.shape))
    center_x = int(np.ceil(center[0]))
    center_y = int(np.ceil(center[1]))
    nb_pixels_x = np.ceil(size_x_mm / voxel_sizes[0])
    nb_pixels_y = np.ceil(size_y_mm / voxel_sizes[1])
    if size_z_mm:
        nb_pixels_z = int(np.ceil(size_z_mm / voxel_sizes[2]))
    # print('Nb of pixels we want in z {}'.format(nb_pixels_z))
    # print('First slice {}'.format(first_slice))
    # check if the crop would be outside the image
    window_x = int(np.ceil(nb_pixels_x / 2))
    window_y = int(np.ceil(nb_pixels_y / 2))
    needz = 0
    if (nb_pixels_z + first_slice) >= size_z:
        extra_pixels = nb_pixels_z + first_slice - size_z
        # last 2.5cm (~5slices) will be mirrored and then background = 0
        if extra_pixels <= 5:
            slice_array = im_array[:, :, :extra_pixels]
            bckg_ext = slice_array[:, :, ::-1]
        elif extra_pixels > 5:
            needz = 1
            slice_array = im_array[:, :, :5]
            slice_array = slice_array[:, :, ::-1]
            slice_bckg = np.zeros((size_x, size_y, (extra_pixels - 5)))
            bckg_ext = np.concatenate((slice_bckg, slice_array), axis=2)
        im_array = np.concatenate((bckg_ext, im_array), axis=-1)
    # print('After z bckg extenstion shape of image {}'.format(im_array.shape))
    size_z = im_array.shape[2]
    # check x
    if (center_x - window_x) < 0:
        bckg_ext = np.zeros((abs(center_x - window_x), size_y, size_z), dtype=np.float32)
        im_array = np.concatenate((bckg_ext, im_array), axis=0)
        center_x = center_x + abs(center_x - window_x)
    elif (center_x + window_x) > size_x:
        bckg_ext = np.zeros((window_x - (size_x - center_x), size_y, size_z), dtype=np.float32)
        im_array = np.concatenate((im_array, bckg_ext), axis=0)
    # print('After x bckg extenstion shape of image {}'.format(im_array.shape))
    size_x = im_array.shape[0]
    # check y
    if (center_y - window_y) < 0:
        bckg_ext = np.zeros((size_x, abs(center_y - window_y), size_z), dtype=np.float32)
        im_array = np.concatenate((bckg_ext, im_array), axis=1)
        center_y = center_y + abs(center_y - window_y)
    elif (center_y + window_y) > size_y:
        bckg_ext = np.zeros((size_x, window_y - (size_y - center_y), size_z), dtype=np.float32)
        im_array = np.concatenate((im_array, bckg_ext), axis=1)
    # print('After y bckg extenstion shape of image {}'.format(im_array.shape))
    size_y = im_array.shape[1]
    im_new = nib.Nifti1Image(im_array, im.affine)
    # print("Ranges x {} - {}".format((center_x - window_x),(center_x + window_x)))
    # print("Ranges y {} - {}".format((center_y - window_y),(center_y + window_y)))
    # print("Ranges z {} - {}".format(-(nb_pixels_z + first_slice-1),-(first_slice-1)))
    crop = im_new.slicer[(center_x - window_x):(center_x + window_x), (center_y - window_y):(center_y + window_y),
           -(nb_pixels_z + first_slice - 1):-(first_slice - 1)]
    return crop, needz


def pad(img, size, axis, background):

    old_size = img.shape[axis]
    pad_size = float(size - old_size) / 2
    pads = [(0, 0), (0, 0), (0, 0)]
    pads[axis] = (math.floor(pad_size), math.ceil(pad_size))
    return np.pad(img, pads, 'constant', constant_values=(background,background ))


def crop(img, size, axis):
    y_min = 0
    y_max = img.shape[0]
    x_min = 0
    x_max = img.shape[1]
    if axis == 0:
        y_min = int(float(y_max - size) / 2)
        y_max = y_min + size
    else:
        x_min = int(float(x_max - size) / 2)
        x_max = x_min + size

    return img[y_min: y_max, x_min: x_max, :]


def crop_or_pad(img, new_size_tuple,background):
    for axis in range(2):
        if new_size_tuple[axis] != img.shape[axis]:
            if new_size_tuple[axis] > img.shape[axis]:
                img = pad(img, new_size_tuple[axis], axis, background)
            else:
                img = crop(img, new_size_tuple[axis], axis)
    return img

#! /usr/bin/env python
# https://github.com/ludwig-ai/ludwig
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



def crop_HN_CT(im_array, center, voxel_sizes, size_x_mm=None, size_y_mm=None, size_z_mm=None, first_slice=2):
    # im_array = im.get_fdata(caching = 'unchanged')
    im_array = im_array.astype(np.int16)
    size_x, size_y, size_z = im_array.shape

    # print('Initial shape of image {}'.format(im_array.shape))

    center_x = int(np.ceil(center[0]))
    center_y = int(np.ceil(center[1]))
    center_z = int(np.ceil(center[2]))

    nb_pixels_x = np.ceil(size_x_mm / voxel_sizes[0])
    nb_pixels_y = np.ceil(size_y_mm / voxel_sizes[1])
    if size_z_mm:
        nb_pixels_z = np.ceil(size_z_mm / voxel_sizes[2])
        window_z = int(np.ceil(nb_pixels_z / 2))
    # check if the crop would be outside the image
    window_x = int(np.ceil(nb_pixels_x / 2))
    window_y = int(np.ceil(nb_pixels_y / 2))

    # window = (window_x, window_y, window_z)
    # print(" Windows {}".format(window))
    needz = 0
    # print("Initial array shape {}".format(im_array.shape))
    if size_z_mm:
        if (center_z - window_z) < 0:
            # bckg_ext = -1024*np.ones((size_x, size_y, abs(center_z - window_z)),dtype=np.float32)
            bckg_ext = np.zeros((size_x, size_y, abs(center_z - window_z)), dtype=np.float32)
            im_array_new = np.concatenate((bckg_ext, im_array), axis=2)
            center_z = center_z + abs(center_z - window_z)
            im_array = im_array_new.copy()
        elif (center_z + window_z) > size_z:
            # bckg_ext = -1024*np.ones((size_x, size_y, (center_z + window_z - size_z)), dtype=np.float32)
            bckg_ext = np.zeros((size_x, size_y, (center_z + window_z - size_z)), dtype=np.float32)
            im_array_new = np.concatenate((im_array, bckg_ext), axis=2)
            im_array = im_array_new.copy()
        # print('After possible z bckg extenstion shape of image {}'.format(im_array.shape))
        size_z = im_array.shape[2]
    # check x
    if (center_x - window_x) < 0:
        # bckg_ext = -1024*np.ones((abs(center_x - window_x), size_y, size_z),dtype=np.float32)
        bckg_ext = np.zeros((abs(center_x - window_x), size_y, size_z), dtype=np.float32)
        im_array_new = np.concatenate((bckg_ext, im_array), axis=0)
        im_array = im_array_new.copy()
        center_x = center_x + abs(center_x - window_x)
    if (center_x + window_x) > size_x:
        # bckg_ext = -1024*np.ones(((center_x + window_x - size_x), size_y, size_z), dtype=np.float32)
        bckg_ext = np.zeros(((center_x + window_x - size_x), size_y, size_z), dtype=np.float32)
        im_array_new = np.concatenate((im_array, bckg_ext), axis=0)
        im_array = im_array_new.copy()
    # print('After possible x bckg extenstion shape of image {}'.format(im_array.shape))
    size_x = im_array.shape[0]
    # check y
    if (center_y - window_y) < 0:
        # bckg_ext = -1024*np.ones((size_x, abs(center_y - window_y), size_z), dtype=np.float32)
        bckg_ext = np.zeros((size_x, abs(center_y - window_y), size_z), dtype=np.float32)
        im_array_new = np.concatenate((bckg_ext, im_array), axis=1)
        center_y = center_y + abs(center_y - window_y)
        im_array = im_array_new.copy()
    if (center_y + window_y) > size_y:
        # bckg_ext = -1024*np.ones((size_x, (center_y + window_y - size_y), size_z), dtype=np.float32)
        bckg_ext = np.zeros((size_x, (center_y + window_y - size_y), size_z), dtype=np.float32)
        im_array_new = np.concatenate((im_array, bckg_ext), axis=1)
        im_array = im_array_new.copy()
    # print('After possible y bckg extenstion shape of image {}'.format(im_array.shape))
    size_y = im_array.shape[1]
    im_array = im_array.astype(np.int16)
    im_new = nib.Nifti1Image(im_array, np.eye(4))
    # print("Ranges x {} - {}".format((center_x - window_x),(center_x + window_x)))
    # print("Ranges y {} - {}".format((center_y - window_y),(center_y + window_y)))
    # print("Ranges z {} - {}".format((center_z - window_z),(center_z + window_z)))
    # crop = im_new.slicer[(center_x - window_x):(center_x + window_x),(center_y - window_y):(center_y + window_y),
    #        -(nb_pixels_z + first_slice-1):-(first_slice-1)]
    if size_z_mm:
        crop = im_new.slicer[(center_x - window_x):(center_x + window_x), (center_y - window_y):(center_y + window_y),
               (center_z - window_z):(center_z + window_z)]
    else:
        crop = im_new.slicer[(center_x - window_x):(center_x + window_x), (center_y - window_y):(center_y + window_y),
               ...]
    return crop, needz


def affine_no_rotation(image):
    initial_affine=image.affine
    initial_affine[0, 0] = abs(initial_affine[0, 0])
    initial_affine[1, 1] = abs(initial_affine[1, 1])
    image = nib.Nifti1Image(np.array(image.get_fdata()), initial_affine)
    return image

def find_volume(array, axis=2):
    sum = 0
    i_largest = 0
    for i in range(array.shape[2]):
        new_sum = array[:, :, i].sum()
        if new_sum >= sum:
            i_largest = i
            sum = new_sum
    if axis == 2:
        new_array = array[:, :, i_largest]
    elif axis == 1:
        new_array = array[:, i_largest, :]
    elif axis == 0:
        new_array = array[i_largest, ...]
    return new_array


def get_seq_data(sequence, ignore_keys):
    seq_data = {}
    for seq in sequence:
        for s_key in seq.dir():
            if s_key in ignore_keys:
                continue
            s_val = getattr(seq, s_key, '')
            if type(s_val) == pydicom.sequence.Sequence:
                _seq = get_seq_data(s_val, ignore_keys)
                seq_data[s_key] = _seq
                continue
            if type(s_val) == str:
                s_val = format_string(s_val)
            else:
                s_val = assign_type(s_val, ignore_keys)
            if s_val:
                seq_data[s_key] = s_val
    return seq_data


def get_df_patient(path_patient, sample_image, path_nifti,
                   tags, exclude_tags, dataset, slice_thickness):
    dicom_dict_tags = search_dicom_tags(sample_image, tags, exclude_tags)
    df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in dicom_dict_tags.items()]))
    df_paths = pd.DataFrame({'PathPatient': [path_patient], 'PathNiftiFile': [path_nifti],
                             'Dataset': dataset, 'SliceThickness': slice_thickness})
    df_patient = pd.concat([df_paths, df], axis=1, join='outer')
    return df_patient


def get_df_slice(path_patient, path_file, ds, tags, exclude_tags, dataset):
    dicom_dict_tags = search_dicom_tags(ds, tags, exclude_tags)
    df = pd.DataFrame.from_dict(dict([(k, pd.Series(v)) for k, v in dicom_dict_tags.items()]))
    df_paths = pd.DataFrame({'PathPatient': [path_patient],
                             'PathImage': [path_file],
                             'Dataset': dataset})
    df_all = pd.concat([df_paths, df], axis=1, join='outer')
    return df_all


def get_suv_factor(sample_image):
    units = sample_image.Units
    error = []
    suv_factor = np.nan
    list_corrections = list(sample_image.CorrectedImage)
    RadiopharmaceuticalStartTime = str(
        sample_image.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime)
    AcquisitionTime = str(sample_image.AcquisitionTime)
    if '.' in RadiopharmaceuticalStartTime:
        RadiopharmaceuticalStartTime = RadiopharmaceuticalStartTime.split(".")[0]
    if '.' in AcquisitionTime:
        AcquisitionTime = AcquisitionTime.split(".")[0]
    if len(RadiopharmaceuticalStartTime) == 5:
        RadiopharmaceuticalStartTime = '0' + RadiopharmaceuticalStartTime
    if len(AcquisitionTime) == 5:
        AcquisitionTime = '0' + AcquisitionTime
    if 'DECY' in list_corrections:
        if units == 'BQML':
            try:
                if sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose != '':
                    dose = float(
                        sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)  # in MBq or Bq??
                else:
                    dose = 0
                HL = float(sample_image.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
                h_start = 3600 * float(RadiopharmaceuticalStartTime[:2])
                h_stop = 3600 * float(AcquisitionTime[:2])
                m_start = 60 * float(RadiopharmaceuticalStartTime[2:4])
                m_stop = 60 * float(AcquisitionTime[2:4])
                s_start = float(RadiopharmaceuticalStartTime[4:6])
                s_stop = float(AcquisitionTime[4:6])
                time2 = (h_stop + m_stop + s_stop - h_start - m_start - s_start)
                activity = dose * (2 ** (-time2 / HL))
                try:
                    weight = float(sample_image.PatientWeight) * 1000
                except:
                    print('Assignig weight of 70kg')
                    weight = 70000.
                suv_factor = weight / activity
            except AttributeError:
                error = 'Attribute to calculate SUV missing'
        elif units == 'GML':
            suv_factor = 1.
        elif units == 'CNTS':
            weight = 1.
            try:
                suv_factor = float(sample_image[0x7053, 0x1000].value)
            except AttributeError:
                try:
                    suv_factor = float(sample_image[0x7053, 0x1009].value) * float(sample_image.RescaleSlope)
                except:
                    error = "Not possible to determine SUV factor - Scan in counts units - Philips tags not accessible"
    else:
        error = 'Units not known'

    return suv_factor, error


def sort_slices(filepaths):
    error = []
    positions = []
    try:
        filepaths.sort(key=lambda x: float(dcmread(x).ImagePositionPatient[2]),
                       reverse=False)
        datasets = [dcmread(x, force=True) for x in filepaths]
        positions = [round(float(ds.ImagePositionPatient[2]), 2) for ds in datasets]
        positions.sort(reverse=False)
        # positions = [round(float(ds.ImagePositionPatient[2]),3) for ds in datasets]
    except AttributeError:
        try:
            filepaths.sort(key=lambda x: float(dcmread(x).SliceLocation), reverse=True)
        except AttributeError:
            try:
                sample_image = dcmread(filepaths[0])
                if sample_image.PatientPosition == 'HFS':
                    filepaths.sort(key=lambda x: dcmread(x, force=True).ImageIndex,
                                   reverse=True)
                if sample_image.PatientPosition == 'FFS':
                    filepaths.sort(key=lambda x: dcmread(x, force=True).ImageIndex)
            except AttributeError:
                error = 'Ordering of slices not possible due to lack of attributes'
    return filepaths, positions, error



def normalizeSUV(ds, nb, UID, suv_factor):
    error = []
    try:
        ds.FrameOfReferenceUID = ds.FrameOfReferenceUID[:2] + UID  # change UID so it is treated as a new image
        ds.SeriesInstanceUID = ds.SeriesInstanceUID[:2] + UID
        ds.SOPInstanceUID = ds.SeriesInstanceUID[:2] + UID + str(nb)
        data_array = ds.pixel_array
        # Check the data type
        bitsRead = str(ds.BitsAllocated)
        sign = int(ds.PixelRepresentation)
        if sign == 1:
            bitsRead = 'int' + bitsRead
        elif sign == 0:
            bitsRead = 'uint' + bitsRead
        data_array = data_array.astype(dtype=bitsRead)  # make sure it is the correct data type
        data16 = (data_array * ds.RescaleSlope + ds.RescaleIntercept) * suv_factor
        data16 = np.array(np.around(data16), dtype=bitsRead)
        ds.RescaleSlope = 1
        ds.RescaleIntercept = 0
        ds.PixelData = data16.tobytes()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
    except AttributeError:
        error = 'Could not get pixel array'
    return ds, error


def normalizeCT(ds, nb, UID):
    error = []
    # try:
    ds.FrameOfReferenceUID = ds.FrameOfReferenceUID[:2] + UID  # change UID so it is treated as a new image
    ds.SeriesInstanceUID = ds.SeriesInstanceUID[:2] + UID
    ds.SOPInstanceUID = ds.SeriesInstanceUID[:2] + UID + str(nb)
    data_array = ds.pixel_array
    data_array[data_array == -2000] = 0
    intercept = ds.RescaleIntercept
    intercept = np.array(intercept, dtype='int16')

    slope = ds.RescaleSlope
    # Check the data type
    bitsRead = str(ds.BitsAllocated)
    sign = int(ds.PixelRepresentation)
    if sign == 1:
        bitsRead = 'int' + bitsRead
        data_array = data_array.astype(dtype=bitsRead)
    elif sign == 0:
        bitsRead = 'uint' + bitsRead
        if intercept < 0:
            data_array = data_array.astype('int16')
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.PixelRepresentation = 1
    if slope != 1:
        print('Slope different than one')
        data_array = slope * data_array
        data_array = data_array.astype('int16')
    if intercept != 0:
        # print(intercept)
        # print(np.min(data_array))
        data_array += intercept
    # data_array[data_array < -1024] = -1024
    # data_array[data_array > 3071] = 3071 #everything above this value is artifact
    # print("Minimum data 16 {}".format(np.min(data_array)))
    # print("Maximum data 16 {}".format(np.max(data_array)))
    data_array[data_array < -1024] = -1024
    data_array[data_array > 3071] = 3071

    data16 = data_array.astype('int16')
    # print("Minimum data 16 {}".format(np.min(data16)))
    # print("Maximum data 16 {}".format(np.max(data16)))
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.PixelData = data_array.tobytes()
    # ds.is_little_endian = True
    ds.is_implicit_VR = True
    # print(ds.is_little_endian)
    # print(ds.is_implicit_VR)
    # except AttributeError:
    #     error = 'Could not get pixel array'
    return ds, error


def format_string(in_string):
    formatted = re.sub(r'[^\x00-\x7f]', r'', str(in_string))  # Remove non-ascii characters
    formatted = ''.join(filter(lambda x: x in string.printable, formatted))
    if len(formatted) == 1 and formatted == '?':
        formatted = None
    return formatted


def assign_type(s, ignore_keys):
    if type(s) == list or type(s) == pydicom.multival.MultiValue:
        try:
            for x in s:
                if type(x) == pydicom.valuerep.DSfloat:
                    list_values = [float(x) for x in s if x not in ignore_keys]
                    return str(tuple(list_values))
                else:
                    list_values = [x for x in s if x not in ignore_keys]
                    return str(tuple(list_values))
        except ValueError:
            try:
                return [float(x) for x in s if x not in ignore_keys]
            except ValueError:
                return [format_string(x) for x in s if ((len(x) > 0) and (x not in ignore_keys))]
    elif type(s) == pydicom.sequence.Sequence:
        return get_seq_data(s, ignore_keys)
    else:
        s = str(s)
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return format_string(s)


def get_mask2(contours, z, Pat_position, pix_spacing, mask):
    pos_r = Pat_position[1]
    spacing_r = pix_spacing[1]
    pos_c = Pat_position[0]
    spacing_c = pix_spacing[0]
    label = mask
    error = 0
    # print("Z {}".format(z))
    for c in contours:
        c_array = np.asarray(c)
        nodes = c_array.reshape((-1, 3))
        # print("Nodes {}".format(nodes))
        # assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
        try:
            z_index = z.index(nodes[0, 2])
        except:
            try:
                c = [round(element, 1) for element in c]
                c_array = np.asarray(c)
                nodes = c_array.reshape((-1, 3))
                z = [round(element, 1) for element in z]
                z_index = z.index(nodes[0, 2])
            except:
                try:
                    c = [round(element) for element in c]
                    c_array = np.asarray(c)
                    nodes = c_array.reshape((-1, 3))
                    z = [round(element) for element in z]
                    z_index = z.index(nodes[0, 2])
                except:
                    error = 1
                    print(z)
                    print(c_array)
                    print("get_mask2 Error")
        # print(z_index)
        if not error:
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = 1
    return label


def find_info(subdir):
    os.chdir(subdir)
    try:
        info = glob.glob('*.{}'.format('csv'))
        info = ''.join(info)
        info_patients = pd.read_csv(info)
        return info_patients
    except:
        try:
            info = glob.glob('*.{}'.format('xlsx'))
            info = ''.join(info)
            info_patients = pd.read_excel(info)
            return info_patients
        except:
            print("No metadata file found in {}".format(subdir))


def build_metadata(metadata_tags, info_patients):
    count_csv = 0
    for tag in metadata_tags:
        try:
            if count_csv == 0:
                metadata_tag = info_patients[[tag]]
                metadata = metadata_tag.copy()
                count_csv = 1
            else:
                metadata = pd.concat([metadata, info_patients[[tag]]], axis=1, sort=True)
                count_csv += 1
        except:
            print('Tag not in info csv file {}'.format(tag))
            if count_csv == 0:
                metadata_tag = pd.DataFrame([np.nan] * len(info_patients.index), columns=[tag])
                metadata = metadata_tag.copy()
                count_csv = 1
            else:
                metadata[tag] = np.nan
                count_csv += 1
    return metadata


def load_metadata(csvfile, df_final, subdir):
    df_new = pd.read_csv(os.path.join(subdir, csvfile))
    for name in df_final.columns:
        if name in df_new.columns:
            df_final[name] = df_new[name]
    return df_final


def find_excel(subdir):
    os.chdir(subdir)
    info = glob.glob('*.{}'.format('xlsx'))
    print(info)
    return info


def search_dicom_tags(ds, tags, ignore_tags):
    dict_dicom = {}
    for name in tags:
        if ((name in ds) and (name not in ignore_tags)):
            dict_dicom[name] = assign_type(ds[name].value, ignore_tags)
        else:
            dict_dicom[name] = 'NaN'
    delete_keys = []
    dict_dicom2 = dict_dicom.copy()
    for key, value in dict_dicom.items():
        if type(value) == dict:
            out_dict = dict_dicom[key]
            dict_dicom2.update(out_dict)
            delete_keys.append(key)
    for key in delete_keys:
        del dict_dicom2[key]
    return dict_dicom2


def get_slice_thickness(path_patient, slice1, slice2):
    slice_thickness = np.nan
    dc1 = dcmread(os.path.join(path_patient, slice1), force=True)
    dc2 = dcmread(os.path.join(path_patient, slice2), force=True)
    error = []
    try:
        slice_thickness = np.abs(dc1.ImagePositionPatient[2] - dc2.ImagePositionPatient[2])
    except AttributeError:
        try:
            slice_thickness = np.abs(dc1.SliceLocation - dc2.SliceLocation)
        except:
            try:
                slice_thickness = dc1.SliceThickness
            except AttributeError:
                slice_thickness = 0
                error = 'Could not determine slice thickness'

    return slice_thickness, error


def get_pixels(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    image = image.astype(np.int16)
    return np.array(image, dtype=np.int16)


def return_tr_day(abrv):
    abrv=abrv.lower()
    if abrv == "1a" or abrv == "2a"or abrv == "3a" or abrv == "1b":
        tr_day="day0"
    elif abrv == "1aa" or abrv == "2aa"or abrv == "3aa" or abrv == "1ba":
        tr_day="day1"
    elif abrv == "1ab" or abrv == "2ab" or abrv == "3ab" or abrv == "1bb":
        tr_day = "day2"
    elif abrv == "1ac" or abrv == "2ac" or abrv == "3ac" or abrv == "1bc":
        tr_day = "day3"
    elif abrv == "1ad" or abrv == "2ad" or abrv == "3ad" or abrv == "1bd":
        tr_day = "day4"
    elif abrv == "1ae" or abrv == "2ae" or abrv == "3ae" or abrv == "1be":
        tr_day = "day5"
    elif abrv == "1af" or abrv == "2af" or abrv == "3af" or abrv == "1bf":
        tr_day = "day6"
    else:
        tr_day = "exception"
    return tr_day