import time
import os
import torch
import numpy as np
from datasets import SIDDDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import mikuti
from mikuti import *
import argparse
from monai.losses import SSIMLoss
import sys
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=6, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--amp", action='store_true', default=False, help="whether to use amp for fast forward")

args = parser.parse_args()
init_environment(args.seed, args.gpu, cudnn=True, benchmark=True, deterministic=False)
ssim_loss = lambda x, y, ws: 1 - SSIMLoss(2, win_size=ws)(x, y)

from core.EPCSCNet import EPCSCNet
model = EPCSCNet(in_ch=3, out_ch=160, elastic_type='impgelu', mean_estimate=True)
state_dict = "./pretrained/EPCSCNet-real-world.pth"
model_name = "EPCSCnet"

model.load_state_dict(torch.load(state_dict, map_location="cpu"))
model = model.cuda()
val_data = SIDDDataset().select('val')
val_loader = DataLoader(val_data, batch_size=5, shuffle=False, num_workers=5)


@torch.cuda.amp.autocast(enabled=args.amp)
def forward(model, LQ, HQ=None):
    loss_info = {}
    loss = 0.0

    rec = model(LQ).clamp(0, 1)

    extra_loss = getattr(model, "extra_loss", None)

    if HQ is not None:
        psnr = mikuti.metric.PSNR(rec, HQ)
        ssim = ssim_loss(rec, HQ, 9).mean()
        loss += torch.nn.functional.mse_loss(rec, HQ)
        loss_info['psnr'] = psnr.item()
        loss_info['ssim'] = ssim.item()

    if extra_loss:
        for key, value in extra_loss.items():
            loss_info[key] = value.item()
            loss += value
        model.extra_loss = None
    loss_info["loss"] = loss.item()
    return rec, loss, loss_info


with torch.no_grad():
    model.eval()
    accumulate = Accumulator()
    start_time = time.time()
    for data in val_loader:
        name = data['name'][0]
        HQ = data['HQ'].cuda().float()
        LQ = data['LQ'].cuda().float()

        rec, loss, loss_info = forward(model, LQ, HQ)
        accumulate.append(loss_info)
        print(name, end=':\t')
        print(format_dict(loss_info))

    loss_mean = accumulate.average()
    loss_mean["time"] = "%.2f min" % ((time.time() - start_time) / 60)
    print(f"#Test mean-> " + format_dict(loss_mean))
