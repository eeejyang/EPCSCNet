import time
import os
import torch
import numpy as np
from datasets import SIDDDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import *
import argparse
from monai.losses import SSIMLoss
from monai.data.dataset import CacheDataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=6, help="random seed")
parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
parser.add_argument("--amp", action='store_true', default=False, help="whether to use amp for fast forward")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--mixup", action='store_true', help="use mixup for data augment")
parser.add_argument("--max_epoch", type=int, default=300, help="maximum epoch")
parser.add_argument("--epoch_step", type=int, default=4000, help="number of steps per epoch")
parser.add_argument("--print_freq", type=int, default=1000, help="verbose frequency")
parser.add_argument("--T0", type=int, default=200, help="T0 for cosine annealing")
parser.add_argument("--Tmul", type=int, default=1, help="Tmul for cosine annealing")

args = parser.parse_args()
init_environment(args.seed, args.gpu, cudnn=True, benchmark=True, deterministic=False)
ssim_loss = lambda x, y, ws: 1 - SSIMLoss(2, win_size=ws)(x, y)

# --------   Configure        ----------------------------------------------------
save_index_keys = {"psnr": -1, "ssim": -1, "loss": 1}

log_path = os.path.join(f"./logs/sidd", time.strftime("%Y-%m%d-%H%M", time.localtime()))

checkpoint_path = f"{log_path}/checkpoint"
tensorboard_path = f"{log_path}/tensorboard"
os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(tensorboard_path, exist_ok=True)
backup_files([__file__, "datasets.py", "core"], log_path)

print(f"PID: {os.getpid()}")
print(f"Training on: SIDD")
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
model = model.cuda()
tbwriter = SummaryWriter(tensorboard_path)
trainset = SIDDDataset()

valset = CacheDataset(trainset.file_dict['val'], trainset.transforms_dict['val'], num_workers=4, cache_rate=1, progress=False)

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=5)
train_loader = endless_generater(train_loader)
val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=5)

# optimizer
global_step = 0
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.Tmul, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()
best_info = {key: float("inf") * order for key, order in save_index_keys.items()}


@torch.cuda.amp.autocast(enabled=args.amp)
def forward(model, LQ, HQ=None):
    loss_info = {}
    loss = 0.0
    rec = model(LQ)
    rec.data.clamp_(0.0, 1.0)
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
    for step in range(args.epoch_step):
        global_step += 1
        data = next(train_loader)
        HQ = data['HQ'].cuda().float()
        LQ = data['LQ'].cuda().float()
        if args.mixup:
            HQ, LQ = mixup_aug(HQ, LQ)
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
