"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from metrics import structural_similarity_index, peak_signal_to_noise_ratio, mean_absolute_error, mean_squared_error
import util.util as util
import torch
import nibabel as nib
import numpy as np
import pandas as pd
from glob import glob
from develop_niftitodicom import convertNsave
import shutil
from pytorch_fid import fid_score

mask_dir="/<PATH>/data/normalization/before_temp/masks"
ct_input_dir="/<PATH>/data/normalization/before_temp/CT_reg"
path_excel_final_split = "/<PATH>/data/excel/train_test_split_second_paper.xls"

threshold_ct_air=-400
threshold_ct_bones=250
ct_max_value=1200
ct_min_value=-1024


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # prepare metrics
    fake_key = 'fake_' + opt.direction[-1]
    real_key = 'real_' + opt.direction[-1]

    res_test = []


    if os.path.exists(path_excel_final_split):
        df_all = pd.read_excel(path_excel_final_split, index_col=0)
        # wb = openpyxl.load_workbook(path_excel_final_split)
        # df_all =pd.ExcelFile(path_excel_final_split)

    results_path=os.path.join("<PATH>/data/latest_test/",opt.name)

    if os.path.exists(results_path):
        print("Such path {} exists. Removing it".format(results_path))
        try:
            shutil.rmtree(results_path)
        except OSError as e:
            print("Error: %s : %s" % (results_path, e.strerror))

    os.makedirs(results_path)

    # metricMAE = metrics.MeanAbsoluteError().to(torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu'))
    # metricMSE = metrics.MeanSquaredError().to(torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu'))


    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break


        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results real_a, fake_B, real_B
        img_path = model.get_image_paths()     # get image paths
        # apply metrics

        #NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        fake_ct=(((visuals["fake_B"] +1) * (1200 + 1024)) / (1 +1)) -1024


        fake_ct_numpy = fake_ct[0].clamp(-1024.0, 1200.0).cpu().float().numpy().astype(np.int16).squeeze()
        # real_ct_numpy = real_ct[0].clamp(-1024.0, 3071.0).cpu().float().numpy().astype(np.int16).squeeze()


        file_name=str(img_path[0].split('/')[-1])
        treatment,slice = file_name.split('-')
        slice= slice.split('.')[0]


        #tbd, cover, if we do not have a mask file, all non background is an image pixels, bigger than -1024
        mask_path=glob(os.path.join(mask_dir, "mask_"+treatment  + '*.nii' ))
        mask_image = nib.load(mask_path[0])
        mask_nii_array = mask_image.get_fdata()
        slice_mask= mask_nii_array[:,:,int(slice)]



        real_slice_path=glob(os.path.join(ct_input_dir, treatment  + '*.nii' ))
        real_ct_image = nib.load(real_slice_path[0])
        real_ct_nii_array = real_ct_image.get_fdata()
        real_ct_nii_array[real_ct_nii_array < ct_min_value] = ct_min_value
        real_ct_nii_array[real_ct_nii_array > ct_max_value] = ct_max_value
        real_ct_numpy= real_ct_nii_array[:,:,int(slice)].astype(np.int16)


        mae = mean_absolute_error(real_ct_numpy, fake_ct_numpy,slice_mask)
        mse = mean_squared_error(real_ct_numpy, fake_ct_numpy,slice_mask)
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
        mae_no_air = mean_absolute_error(real_ct_numpy, fake_ct_numpy, mask_air_joined)

        ######get masked bone regions, for metrics calcs

        mask_bones = np.zeros(real_ct_numpy.shape)
        mask_bones[real_ct_numpy > threshold_ct_bones] = 1
        mask_bones[real_ct_numpy <= threshold_ct_bones] = 0

        mae_bones = mean_absolute_error(real_ct_numpy, fake_ct_numpy, mask_bones)

        res_test.append([mae, mse, psnr,  ssim, mae_no_air, mae_bones])


        #save dicoms and niftis
        #fake
        path_fake = os.path.join(results_path,"fake",treatment)
        os.makedirs(path_fake, exist_ok=True)

        fake_ct_numpy=np.rot90(fake_ct_numpy,-1)
        fake_ct_numpy=np.fliplr(fake_ct_numpy).astype(np.int16)
        convertNsave(fake_ct_numpy,  path_fake,  df_all, treatment, slice)

        path_fake_nifti = os.path.join(results_path, "fake_nifti")
        os.makedirs(path_fake_nifti, exist_ok=True)
        path_slice_fake=os.path.join(path_fake_nifti,treatment+"_"+  slice+'.nii')
        fake_ct_nif = nib.Nifti1Image(fake_ct_numpy, np.eye(4))
        nib.save(fake_ct_nif, path_slice_fake)



        #real
        path_real= os.path.join(results_path,"real",treatment)
        os.makedirs(path_real, exist_ok=True)


        real_ct_numpy=np.rot90(real_ct_numpy,-1)
        real_ct_numpy=np.fliplr(real_ct_numpy).astype(np.int16)
        convertNsave(real_ct_numpy, path_real, df_all, treatment, slice)

        path_real_nifti = os.path.join(results_path, "real_nifti")
        os.makedirs(path_real_nifti, exist_ok=True)
        path_slice_real=os.path.join(path_real_nifti,treatment+"_"+ slice+'.nii')
        real_ct_nif = nib.Nifti1Image(real_ct_numpy, np.eye(4))
        nib.save(real_ct_nif, path_slice_real)


    #     if i % 5 == 0:  # save images to an HTML file
    #         print('processing (%04d)-th image... %s' % (i, img_path))
    #     save_images(webpage, visuals, img_path, width=opt.display_winsize)
    # webpage.save()  # save the HTML


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





