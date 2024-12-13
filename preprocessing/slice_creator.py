import os
import numpy as np
import pandas as pd
import nibabel as nib
global path_all
from glob import glob
import shutil
import imageio


path_normalized_MR = "<PATH_TO_DATA>/normalization/after_npeaks/MR_bfc_2x"
path_normalized_CT = "<PATH_TO_DATA>/normalization/CT_reg_norm_temp_air_overwrite"



# for nn_input_mode in ["2d","pseudo3d"]:
for nn_input_mode in [ "2d"]:
    # # #save in the aligned and not aligned fashion
    path_model = "<PATH_TO_DATA>/2nd_paper_nyul_AIROR_model_test_" + nn_input_mode
    path_pix = "<PATH_TO_DATA>/2nd_paper_nyul_AIROR_pix2pix_test_" + nn_input_mode

    path_excel_final_split = "<PATH_TO_DATA>/excel/train_test_split_second_paper.xlsx"

    accepted_tr_days= ["day0"]
    accepted_modal_folders= ["CT","MR"]
    # extension="nii" # "png" or "nii"
    extension="png"


    # removing folders with files if they exist and create new empty folders
    for path_r in [path_model, path_pix]:
        if os.path.exists(path_r):
            print("!IMPORTANT! path={} exists. Removing it with all files inside".format(path_r))
            try:
                shutil.rmtree(path_r)
            except OSError as e:
                print("Error: %s : %s" % (path_r, e.strerror))
        os.makedirs(path_r)

    # create train and test dirs for pix and cycle\cut
    os.makedirs( os.path.join(path_model,"full", "train", "trainA"))
    os.makedirs( os.path.join(path_model,"full", "train", "trainB"))
    os.makedirs(os.path.join(path_model, "full","test", "testA"))
    os.makedirs(os.path.join(path_model, "full","test", "testB"))

    os.makedirs( os.path.join(path_pix,"full", "A", "train"))
    os.makedirs( os.path.join(path_pix,"full", "A", "test"))
    os.makedirs( os.path.join(path_pix,"full", "B", "train"))
    os.makedirs( os.path.join(path_pix,"full", "B", "test"))

    os.makedirs( os.path.join(path_model,"validation", "train", "trainA"))
    os.makedirs( os.path.join(path_model,"validation", "train", "trainB"))
    os.makedirs(os.path.join(path_model, "validation","test", "testA"))
    os.makedirs(os.path.join(path_model, "validation","test", "testB"))

    os.makedirs( os.path.join(path_pix,"validation", "A", "train"))
    os.makedirs( os.path.join(path_pix,"validation", "A", "test"))
    os.makedirs( os.path.join(path_pix,"validation", "B", "train"))
    os.makedirs( os.path.join(path_pix,"validation", "B", "test"))


    if os.path.exists(path_excel_final_split):
        df_all = pd.read_excel(path_excel_final_split, index_col=0)

        for index, line in df_all.iterrows():
            if line.TreatmentDay in accepted_tr_days and line.ModalityFolder in accepted_modal_folders and line.Saved_nifti!="0":

                # for split_mode in ["full","validation"]:
                for split_mode in ["full"]:
                    if split_mode=="full":
                        split=line.Split
                    elif split_mode=="validation" and line.Split_incl_val=="val":
                        split="test"
                    elif split_mode == "validation" and line.Split_incl_val == "train":
                        split = line.Split_incl_val
                    elif split_mode == "validation" and line.Split_incl_val == "test":
                        continue



                    if line.ModalityFolder=="MR":
                        nifti_norm_folder=path_normalized_MR
                        newdir_slices_path_pix=os.path.join(path_pix,split_mode, "A", split)
                        newdir_slices_path_unpaired =os.path.join(path_model,split_mode, split, split+ "A")
                    else:
                        nifti_norm_folder = path_normalized_CT
                        newdir_slices_path_pix = os.path.join(path_pix,split_mode, "B", split)
                        newdir_slices_path_unpaired = os.path.join(path_model,split_mode,split, split + "B")


                    image_paths = glob(os.path.join(nifti_norm_folder,line.Folder+"*.nii") )
                    if image_paths is not None:
                        for image_path in image_paths:

                            if os.path.exists(image_path):

                                image=nib.load(image_path)
                                sample = np.array(image.get_fdata())

                                for i in range(1,sample.shape[2]-1):

                                    if nn_input_mode=="2d":
                                        im = image.slicer[..., i:(i + 1)]
                                    else:
                                        im = image.slicer[..., (i-1):(i + 2)]

                                    path_nifti_slice_pix = os.path.join(newdir_slices_path_pix,  line.Folder+ "-" +str(i) + "."+extension)
                                    path_nifti_slice_unpaired = os.path.join(newdir_slices_path_unpaired,line.Folder+ "-" +str(i) + "."+extension)

                                    im_data = im.get_fdata(caching="unchanged")

                                    # handling outliers after Nyul norm of MRI
                                    im_data[im_data < 0] = 0
                                    im_data[im_data > 1] = 1

                                    im = nib.Nifti1Image(im_data, image.affine)

                                    for path_save in [path_nifti_slice_pix,path_nifti_slice_unpaired ]:
                                        if os.path.exists(path_save):
                                            os.remove(path_save)
                                        print(path_save)

                                        if extension=="png":
                                            im_data = im_data * 255
                                            # im_data = im_data.astype(np.int16)
                                            imageio.imwrite(path_save, im_data)
                                        else:
                                            nib.save(im, path_save)

                            else:
                                print("!IMPORTANT! path not exists ={}".format(image_path))

print ("finished")