# MR-based synthetic CT generation towards MR-only radiotherapy

## Clone our repository
- Clone this repository:
```bash
git clone https://github.com/medical-physics-usz/synthetic_CT_generation
cd sCT_generation
```

## Set up environment
1)  ```conda env create --file sCT_env.yml ```
2)  ```conda env create --file sCT_pytorch_env.yml ```
3) If the intensity normalization package has not been installed correctly:
 ```
pip install intensity-normalization
 ```
or check instructions here https://github.com/jcreinhold/intensity-normalization
<TBD: raise a bug: Nyul is not working if we have slices with the only backgorund in the set>

## Preprocessing
1) Script CT_MR_preprocessing.py transforms the series of DICOM files to a single NIFTI file for every modality. Then, it creates a set of masks: 
* extraction of body mask for non-registered CTs is based on the background filtering: threshold_ct_body_mask = -400 + max contouring and morphological operations
* extraction of body mask for registered CTs\MRs  is based on the delineated structure file with the tag ['skin'] +morphological operations for smoothing
* extraction of delineated set of organs for tissue-based intensity normalisation (based on STRUCT: ['liver'],['water','bladder','gallbladder', 'water_or'] ) + (“fat” extracted from CT_reg based on HU: pixels with intensities in [-160;-60])
* extraction of tumour mask with the biggest delineated volume out of following tags ['ptv2_vX_1a'], or ['ptv1_vX_xa'] and take the biggest
* creates NIFTIs for every treatment and modality with all background pixels set to 0 or -1024
* check cmd parameters
 ```
!python -u CT_MR_preprocessing.py "$@"
 ```
2) Script resampling.py resize the voxels and crop_or_pad image as following:
* accepted_modal_folders= ["CT_reg","MR"]
* accepted_tr_days= ["day0"]
* crop_x_desired=256
* crop_y_desired=256
* desired_voxel_size_x = 1.6304
* desired_voxel_size_y = 1.6304
* desired_voxel_size_z =3
* background_mri=0
* background_ct=-1024
* check cmd parameters
 ```
!python -u resampling.py "$@"
 ```
 Then, it saves only the 20 slides around tumour_center_slide for each image. In case it is close to the image border, it moves the 20-slices window to another direction in order to get not corrupted slices. Exceptions: 3 patients with diff shapeZ of CT_reg and MR -> then, defined manually. <TBD>All masks resampled respectively and saved 
3) Script ct_min_max_normalization.py performs min-max normalization of prepared registered CTs
 ```
!python -u /ct_min_max_normalization.py "$@"
 ```
4) Script mry_norm_nyul performs Nyul, intensity based, non-linear normaliztion of MRIs
 ```
nyul-normalize data/normalization/before/MR/ -m data/normalization/before/masks/ -o data/normalization/Nyul_normalized/ -v

 ```
5) Script slice_creator.py create axial slices out of both normalized modalities and writes them in folders for alligned and not-alligned dataset training
 ```
python -u slice_creator.py "$@"
 ```
6) We have pix2pix, cycleGAN and CUT, which could be trained on NIFTIs <TBD>: check the input and output ranges of all NN
7) NN script test.py test the network performance, based on the provided slices, saves the outputs in nifti format + calculate the geometrical metrics (MAE, MSE, PSNR, SSIM ) for the pixels in the body contour
8) Script niftitodicom.py creates dicom slices from the NN nifti outputs for CT_reg in HU <TBD>: check for MR,what is there that could be used further for the dosimetric accuracy calcs
 ```
python -u /CUT/niftitodicom.py "$@"
 ```

## Train the models
Change directory:
 ```bash
cd CUT
```
 
 
### pix2pix
 For pix2pix the alligned dataset firstly should be created, then, the model could be trained
 ```bash
 !python datasets/combine_A_and_B.py --fold_B data/pix2pix_test_scr/A --fold_A data/pix2pix_test_scr/B --fold_AB data/pix2pix_test_scr/data_AB

!python train.py --dataroot data/pix2pix_test_nifti/data_AB --name nifti_NEW_sCT_CUT_whole_data_pix2pix --model pix2pix --direction BtoA --input_nc 1 --output_nc 1
 ```

### CycleGAN
 ```
!python train.py --dataroot 'data/model_test_nifti/train' --name nifti_New_sCT_CUT_cyclegan --model cycle_gan --input_nc 1 --output_nc 1 ```
  ```
 
 ### CUT
 ```bash
!python train.py --dataroot '/data/model_test_nifti/train' --name nifti_sCT_CUT_cut --model cut --CUT_mode CUT --input_nc 1 --output_nc 1
 ```


## Test the models
To obtain the synthetic CT images, the steps to reproduce are the following:

### pix2pix
 For pix2pix the alligned dataset firstly should be created, then, the model could be trained
 ```
!python test.py --phase test --dataroot data/pix2pix_test_nifti/data_AB --name nifti_NEW_sCT_CUT_whole_data_pix2pix --model pix2pix --direction BtoA --input_nc 1 --output_nc 1
 ```
<TBD:change direction of training> 

### CycleGAN
 ```
!python test.py --dataroot 'data/model_test_nifti/test' --phase test --name nifti_New_sCT_CUT_cyclegan --model cycle_gan --input_nc 1 --output_nc 1
  ```
 
 ### CUT
 ```bash
!python test.py --dataroot 'data/model_test_nifti/test' --phase test  --name nifti_sCT_CUT_cut --model cut --CUT_mode CUT --input_nc 1 --output_nc 1
 ```
After testing is completed, the results will be shown under the results folder in the repository directory

## Calculate DVH differences: dCT-sCT
Install MatRAD from: https://github.com/e0404/matRad
Replace corresponding files from the DVH calculation repo and pass your parameters for plan calcs and tumour\OAR structure selection
