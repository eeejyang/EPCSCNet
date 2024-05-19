import time
import os
import torch
import numpy as np
from datasets import GaussianNoiseDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
from monai.losses import SSIMLoss
from PIL import Image
import sys
import pandas as pd
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=6, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--amp", action='store_true', default=False, help="whether to use amp for fast forward")
parser.add_argument("--color", action='store_true', default=False, help="whether to convert to grayscale image")
parser.add_argument("--sigma", type=int, default=50, help="gaussion noise level")

args = parser.parse_args()
init_environment(args.seed, args.gpu, cudnn=True, benchmark=True, deterministic=False)
ssim_loss = lambda x, y, ws: 1 - SSIMLoss(2, win_size=ws)(x, y)
color_type = "color" if args.color else "gray"

from core.EPCSCNet import EPCSCNet
model = EPCSCNet(in_ch=3 if args.color else 1, out_ch=160, elastic_type='impgelu', mean_estimate=False)

state_dict = torch.load(f"./pretrained/EPCSCNet-{color_type}-gaussian.pth", map_location="cpu")[f'{args.sigma}']
model.load_state_dict(state_dict)
model.cuda()

val_path = ["./datasets/CBSD68"]
valset = GaussianNoiseDataset(val_path, args.sigma, args.color, train=False)
val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=5)
result_dir = "results"
os.makedirs(f"{result_dir}/imgs", exist_ok=True)


@torch.cuda.amp.autocast(enabled=args.amp)
def forward(model, LQ, HQ=None):
    loss_info = {}
    loss = 0.0

    rec = model(LQ).clamp(0, 1)
    extra_loss = getattr(model, "extra_loss", None)

    if HQ is not None:
        psnr = PSNR(rec, HQ)
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
    accumulate = Accumulator(cache=True)
    start_time = time.time()
    for data in val_loader:
        name = data['name'][0]
        HQ = data['HQ'].cuda().float()
        LQ = data['LQ'].cuda().float()

        rec, loss, loss_info = forward(model, LQ, HQ)
        loss_info['name'] = name
        accumulate.append(loss_info)
        print(format_dict(loss_info))

        rec = rec.moveaxis(1, -1).squeeze().clamp(0, 1).mul(255).round().cpu().numpy().astype("uint8")
        Image.fromarray(rec).save(f"{result_dir}/imgs/{name}")

    loss_mean = accumulate.average()
    loss_mean["time"] = "%.2f min" % ((time.time() - start_time) / 60)
    print(f"#Test mean-> " + format_dict(loss_mean))

    info_table = pd.DataFrame(accumulate.dict_cache)
    info_table.to_csv(f"{result_dir}/eval_info.csv", sep=',', float_format="%.6f", index=None)
