import time
import os
import torch
import numpy as np
import argparse
import sys
import scipy.io as sio
import h5py
import glob
from utils import *

root_dir = "./datasets/DND"

sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=6, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--amp", action='store_true', default=False, help="whether to use amp for fast forward")

args = parser.parse_args()
init_environment(args.seed, args.gpu, cudnn=True, benchmark=True, deterministic=False)


from core.EPCSCNet import EPCSCNet
model = EPCSCNet(in_ch=3 if args.color else 1, out_ch=160, elastic_type='impgelu', mean_estimate=True)
state_dict = "./pretrained/EPCSCNet-real-world.pth"
model_name = "ECSCnet"

model.load_state_dict(torch.load(state_dict, map_location="cpu"))
model = model.cuda()

out_folder = f"./submit/dnd/{model_name}"
os.makedirs(out_folder, exist_ok=True)

img_list = sorted(glob.glob(f"{root_dir}/images_srgb/*.mat"))
info = h5py.File(f"{root_dir}/info.mat", mode='r')['info']

bboxes = info['boundingboxes']

with torch.no_grad():
    for idx in range(len(img_list)):
        filename = img_list[idx].split('/')[-1]
        img = h5py.File(img_list[idx], mode='r')['InoisySRGB']
        img = np.array(img, dtype='float32').T
        ref = bboxes[0][idx]
        boxes = np.array(info[ref]).T
        Idenoised = np.zeros((20,), dtype='object')
        for k in range(20):
            pos = [int(boxes[k, 0] - 1), int(boxes[k, 2]), int(boxes[k, 1] - 1), int(boxes[k, 3])]
            img_crop = img[pos[0] : pos[1], pos[2] : pos[3], :].copy()
            H, W = img_crop.shape[:2]
            img_crop = torch.from_numpy(img_crop).float().cuda()
            img_crop = img_crop.moveaxis(-1, 0)[None, ...].contiguous()

            denoised_patch = model.forward(img_crop).clip(min=0, max=1)

            denoised_patch = denoised_patch.cpu().numpy().squeeze()
            denoised_patch = np.transpose(denoised_patch, [1, 2, 0]).astype("float32")
            Idenoised[k] = denoised_patch.copy()
            # save denoised data
            print('%s crop %d/%d' % (filename, k + 1, 20))
        sio.savemat(
            f"{out_folder}/{idx+1:04d}.mat",
            {'Idenoised': Idenoised, "israw": False, "eval_version": "1.0"},
        )

        print('[%d/%d] %s done\n' % (idx + 1, 50, filename))
