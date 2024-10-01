

import nibabel
import numpy as np
import os
import shutil
from glob import glob
from pydicom import read_file, dcmread
import pandas as pd
import math
from nibabel import processing
import scipy



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

def resize_data_volume_by_scale(data, scale):
   """
   Resize the data based on the provided scale
   """
   # scale_list = [scale,scale,scale]
   return scipy.ndimage.interpolation.zoom(data, scale, order=0)


def crop_or_pad(img, new_size_tuple,background):
    for axis in range(2):
        if new_size_tuple[axis] != img.shape[axis]:
            if new_size_tuple[axis] > img.shape[axis]:
                img = pad(img, new_size_tuple[axis], axis, background)
            else:
                img = crop(img, new_size_tuple[axis], axis)
    return img


def resample(image, spacing, new_spacing):
    # Determine current pixel spacing
    print(image.shape)
    spacing= np.array(list(spacing))
    new_spacing = np.array(list(new_spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=0)

    return image, new_spacing






###TBD: change isocenter, rotations and the dimensionality weight\height 1.63 256


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

## this function is based on https://github.com/amine0110/nifti2dicom
#TBD check correctness of all tags in MR mode
def convertNsave(arr, file_dir,  df_all, treatment, slice, mode="CT", path_all = "<PATH>/data/data_exported"):
    """
    `arr`: parameter will take a numpy array that represents only one slice.
    `file_dir`: parameter will take the path to save the slices
    `index`: parameter will represent the index of the slice, so this parameter will be used to put 
    the name of each slice while using a for loop to convert all the slices
    """

    pat_line=df_all.loc[(df_all['Folder'] == str(treatment)) & (df_all['ModalityFolder'] == mode)]
    print("slice and first_slice_in_ring")
    print(slice)
    print(pat_line['first_slice_in_ring'])

    slice_nbr= int(slice) + int(pat_line['first_slice_in_ring'])
    if mode=="CT":
        modality_dicom_path = glob(os.path.join(path_all,treatment + "*", "CT_reg"))
    elif mode=="MR":
        modality_dicom_path = glob(os.path.join(path_all, treatment + "*","Plan", "MR"))
        exemplary_CT=dcmread('<PATH>/exemplary_CT/CT1.3.12.2.1107.5.1.4.65204.30000020121008484958000003752.dcm')
    else:
        modality_dicom_path=[]
        print("exception! Choose another mode")

    if os.path.exists(modality_dicom_path[0]):
        dcm_files = [file for file in os.listdir(modality_dicom_path[0]) if file.startswith(mode)]
        dcm_paths = [os.path.join(modality_dicom_path[0], file) for file in dcm_files if
                     os.path.isfile(os.path.join(modality_dicom_path[0], file))]
        # dicom arguments
        sorted_dcm_paths, z_coords, error_sorting = sort_slices(dcm_paths)
        # print(sorted_dcm_paths[slice_nbr])
        dicom_file = dcmread(sorted_dcm_paths[slice_nbr])
    else:
        dicom_file = dcmread('<PATH>/exemplary_CT/CT1.3.12.2.1107.5.1.4.65204.30000020121008484958000003752.dcm')


    #   tags RescaleSlope and RescaleIntercept handling
    arr=arr+1024

    pixel_size = np.array(dicom_file.PixelSpacing)
    # .replace("[", "").replace("]", "").replace(",", "").split()).astype(np.float32)
    current_dim_x = int(pixel_size[0] * 10000) / 10000
    current_dim_y = int(pixel_size[1] * 10000) / 10000

    desired_size_rows=dicom_file.Rows
    desired_size_columns=dicom_file.Columns


    if current_dim_x != 1.6304 or current_dim_y != 1.6304:

        arr,sp=resample(arr,[1.6304,1.6304], [dicom_file.PixelSpacing[0],dicom_file.PixelSpacing[1]])
        print("shape")
        print(arr.shape)

    background = 0
    # for patients with displaced center of mass, adjust it
    if treatment in ["PAT_LIST"]:
        arr = np.pad(arr, ((30, 0), (0, 0)), 'constant', constant_values=(background, background))
        # nii_array_resampled = nii_array_resampled[30:, :, :]

    elif treatment in ["PAT_LIST"]:
        # nii_array_resampled = nii_array_resampled[5:, :, :]
        arr = np.pad(arr, ((5, 0), (0, 0)), 'constant', constant_values=(background, background))

    elif treatment in ["PAT_LIST"]:
        # nii_array_resampled = nii_array_resampled[15:, :, :]
        arr = np.pad(arr, ((15, 0), (0, 0)), 'constant', constant_values=(background, background))

    elif treatment == "PAT_LIST":
        # nii_array_resampled = nii_array_resampled[20:, :-20, :]
        arr = np.pad(arr, ((20, 0), (0, 20)), 'constant', constant_values=(background, background))

    elif treatment in ["PAT_LIST"]:
        # nii_array_resampled = nii_array_resampled[:-20, :, :]
        arr = np.pad(arr, ((0, 20), (0, 0)), 'constant', constant_values=(background, background))

    if arr.shape[0] != desired_size_rows or arr.shape[1] != desired_size_columns:
        arr = np.expand_dims(arr, axis=2)
        arr = crop_or_pad(arr, (desired_size_rows, desired_size_columns, 1), background)
        arr = arr.squeeze().astype('uint16')
        if mode == "CT":
            dicom_file.Rows = desired_size_rows
            dicom_file.Columns = desired_size_columns
        else:
            exemplary_CT.Rows = desired_size_rows
            exemplary_CT.Columns = desired_size_columns
        new_name = treatment + "-" + str(slice)


    if mode == "CT":
        dicom_file.PixelData = arr.tobytes()
        dicom_file.RescaleSlope =1
        dicom_file.RescaleIntercept=-1024
        dicom_file.save_as(os.path.join(file_dir, new_name + '.dcm'))

    else:
        exemplary_CT.PixelData = arr.tobytes()
        exemplary_CT.RescaleSlope =1
        exemplary_CT.RescaleIntercept=-1024
        exemplary_CT.InstanceNumber = dicom_file.InstanceNumber
        exemplary_CT.SOPInstanceUID=dicom_file.SOPInstanceUID
        exemplary_CT.InstanceCreationDate=dicom_file.InstanceCreationDate
        exemplary_CT.InstanceCreationTime = dicom_file.InstanceCreationTime
        exemplary_CT.ImagePositionPatient=dicom_file.ImagePositionPatient
        exemplary_CT.ImageOrientationPatient=dicom_file.ImageOrientationPatient
        exemplary_CT.SliceLocation=dicom_file.SliceLocation
        exemplary_CT.save_as(os.path.join(file_dir, new_name + '.dcm'))



def nifti2dicom_1file(nifti_dir,file_name, out_dir, df_all):
    """
    This function is to convert only one nifti file into dicom series
    `nifti_dir`: the path to the one nifti file
    `out_dir`: the path to output
    """

    nifti_file = nibabel.load(nifti_dir)
    nifti_array = nifti_file.get_fdata()
    treatment, slice = [file_name.split('_')[:-1],file_name.split('_')[-1]]
    treatment="_".join(treatment)
    slice = int(slice.split('.')[0])


    convertNsave(nifti_array, out_dir, df_all, treatment, slice)


def nifti2dicom_mfiles(nifti_dir, out_dir,df_all):
    """
    This function is to convert multiple nifti files into dicom files
    `nifti_dir`: You enter the global path to all of the nifti files here.
    `out_dir`: Put the path to where you want to save all the dicoms here.
    PS: Each nifti file's folders will be created automatically, so you do not need to create an empty folder for each patient.
    """

    files = os.listdir(nifti_dir)
    if os.path.exists(out_dir):
        print("!IMPORTANT! out_dir {} could exist. Removing it with all files inside".format(out_dir))
        try:
            shutil.rmtree(out_dir)
        except OSError as e:
            print("Error: %s : %s" % (out_dir, e.strerror))

    os.makedirs(out_dir)

    for file in files:
        in_path = os.path.join(nifti_dir, file)
        nifti2dicom_1file(in_path,file, out_dir, df_all)

