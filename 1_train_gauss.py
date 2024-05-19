import time
import os
import torch
import numpy as np
from datasets import GaussianNoiseDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse
from monai.losses import SSIMLoss
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=6, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--amp", action='store_true', default=False, help="whether to use amp for fast forward")
parser.add_argument("--color", action='store_true', default=False, help="whether to convert to grayscale image")
parser.add_argument("--sigma", type=int, default=50, help="gaussion noise level")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--max_epoch", type=int, default=300, help="maximum epoch")
parser.add_argument("--print_freq", type=int, default=1000, help="verbose frequency")

args = parser.parse_args()
init_environment(args.seed, args.gpu, cudnn=True, benchmark=True, deterministic=False)
ssim_loss = lambda x, y, ws: 1 - SSIMLoss(2, win_size=ws)(x, y)

# --------   Configure        ----------------------------------------------------
save_index_keys = {"psnr": -1, "ssim": -1, "loss": 1}
if args.color:
    log_path = os.path.join(f"./logs/color-{args.sigma}", time.strftime("%Y-%m%d-%H%M", time.localtime()))
else:
    log_path = os.path.join(f"./logs/gray-{args.sigma}", time.strftime("%Y-%m%d-%H%M", time.localtime()))

checkpoint_path = f"{log_path}/checkpoint"
tensorboard_path = f"{log_path}/tensorboard"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)
backup_files([__file__, "datasets.py", "core"], log_path)

print(f"PID: {os.getpid()}")
print(f"Training on: Gaussian noise")
print(f"Logging path: {log_path}")
print()

print("---------------------------- Parameters -----------------------------------------")
print(format_dict(vars(args), ncols=8))
print("---------------------------------------------------------------------------------")
print()


# fmt: off
# -------------------------------------------------------------------------------

from core.EPCSCNet import EPCSCNet
model = EPCSCNet(in_ch=3 if args.color else 1, out_ch=160, elastic_type='impgelu', mean_estimate=False)

# fmt: on
# ---------------------------------------------------------------------------------

train_path = ["/home/yanglj/Desktop/datasets/CBSD432", "/home/yanglj/Desktop/datasets/waterloo"]
val_path = ["/home/yanglj/Desktop/datasets/CBSD68"]

tbwriter = SummaryWriter(tensorboard_path)
trainset = GaussianNoiseDataset(train_path, args.sigma, args.color, train=True)
valset = GaussianNoiseDataset(val_path, args.sigma, args.color, train=False)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=5)
val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=5)

# optimizer
global_step = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
scaler = torch.cuda.amp.GradScaler()
best_info = {key: float("inf") * order for key, order in save_index_keys.items()}


@torch.cuda.amp.autocast(enabled=args.amp)
def forward(model, LQ, HQ=None):
    loss_info = {}
    loss = 0.0
    rec = model(LQ)
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


def train_one_epoch(epoch):
    accumulate = Accumulator()
    global global_step
    model.train()
    start_time = time.time()
    for step, data in enumerate(train_loader):
        global_step += 1
        HQ = data['HQ'].cuda().float()
        LQ = data['LQ'].cuda().float()

        rec, loss, loss_info = forward(model, LQ, HQ)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accumulate.append(loss_info)
        if step % args.print_freq == 0:
            info = f"--- |  train {epoch:04d}/{step:04d}-> " + format_dict(loss_info)
            print(info)

    scheduler.step()
    loss_mean = accumulate.average()
    loss_mean['lr'] = optimizer.param_groups[0]['lr']
    tbadd_dict(tbwriter, loss_mean, global_step, group='train')
    loss_mean['time'] = "%.2f min" % ((time.time() - start_time) / 60)
    print(f"*train epoch {epoch:04d} mean-> " + format_dict(loss_mean))


@torch.no_grad()
def evaluate(epoch):
    model.eval()
    accumulate = Accumulator()
    start_time = time.time()
    for data in val_loader:
        HQ = data['HQ'].cuda().float()
        LQ = data['LQ'].cuda().float()

        rec, loss, loss_info = forward(model, LQ, HQ)
        accumulate.append(loss_info)

    loss_mean = accumulate.average()
    tbadd_dict(tbwriter, loss_mean, epoch, group='val')
    loss_mean["time"] = "%.2f min" % ((time.time() - start_time) / 60)
    print(f"#Test  epoch {epoch:04d} mean-> " + format_dict(loss_mean))
    return loss_mean


for epoch in range(args.max_epoch):
    train_one_epoch(epoch)
    loss_info = evaluate(epoch)
    checkpoints = dict(
        epoch=epoch,
        net=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
    )

    torch.save(checkpoints, f"{checkpoint_path}/recent.ckpt")
    for key, order in save_index_keys.items():
        cur = loss_info[key]
        prev = best_info[key]
        if cur * order < prev * order:
            best_info[key] = cur
            torch.save(checkpoints, f"{checkpoint_path}/best_{key}.ckpt")
            print(f"Best {key}! epoch {epoch:04d}: previous = {prev:.6f} current = {cur:.6f}")
    print()

tbwriter.close()
