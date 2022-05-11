

"""
## Learning Enriched Features for Real Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## ECCV 2020
## https://arxiv.org/abs/2003.06792
"""


import numpy as np
import os
import argparse
from tqdm import tqdm

import paddle.nn as nn
import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F

import scipy.io as sio
from networks.MIRNet_model import MIRNet
from dataloaders.data_rgb import get_validation_data
import utils
from skimage import img_as_ubyte

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='./SIDD_patches/val/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/denoising/sidd/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/model_denoising.pdparams',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=16, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)


model_restoration = MIRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)


# model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()


with paddle.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0]
        rgb_noisy = data_test[1]
        filenames = data_test[2]
        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = paddle.clip(rgb_restored,0,1)

        tmp_psnr = utils.batch_PSNR(rgb_restored, rgb_gt, 1.)
        print(tmp_psnr)
        psnr_val_rgb.append(tmp_psnr)

        rgb_gt = rgb_gt.transpose([0, 2, 3, 1]).numpy()
        rgb_noisy = rgb_noisy.transpose([0, 2, 3, 1]).numpy()
        rgb_restored = rgb_restored.transpose([0, 2, 3, 1]).numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)
            
psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb))

