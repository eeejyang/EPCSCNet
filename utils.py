import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import pickle as pkl
import random
import torch.distributed as distributed
import os
import shutil


def PSNR(x: torch.Tensor, y: torch.Tensor, reduce=True):
    mse = torch.pow((x - y), 2).mean(dim=[-1, -2, -3])
    ret = -10 * torch.log10(mse)

    if reduce:
        ret = ret.mean()
    return ret


def backup_files(path_list, destination):
    for path in path_list:
        target = os.path.join(destination, os.path.basename(path))
        if os.path.isdir(path):
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy(path, target, follow_symlinks=True)


def init_environment(seed=None, device=0, cudnn=True, benchmark=True, deterministic=False):
    torch.cuda.set_device(device)
    torch.backends.cudnn.enabled = cudnn
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# data
def endless_generater(loader):
    while True:
        for data in loader:
            yield data


def mixup_aug(*img_args: torch.Tensor):
    assert len(img_args) > 0
    B = img_args[0].size(0)
    device = img_args[0].device
    indices = torch.randperm(B)
    dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))
    lam = dist.rsample((B, 1)).view(-1, 1, 1, 1).to(device)
    output = []
    for img in img_args:
        shuffle = img[indices]
        mixed = lam * img + (1 - lam) * shuffle
        output.append(mixed)
    output = output[0] if len(output) == 1 else output
    return output


def RGB2YCbCr(img):

    r, g, b = torch.split(img, 1, dim=1)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    img = torch.cat([y, cb, cr], dim=1)
    return img


def YCbCr2RGB(img):
    y, cb, cr = torch.split(img, 1, dim=1)

    r = y + 1.403 * (cr - 0.5)
    g = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
    b = y + 1.773 * (cb - 0.5)
    img = torch.cat([r, g, b], dim=1).clamp(min=0.0, max=1.0)
    return img


class Accumulator:

    def __init__(self, cache=False) -> None:
        super().__init__()
        self.cnt = 0
        self.dict_sum = {}
        self.dict_cache = {}
        self.cache = cache

    def append(self, data: dict):
        for key, value in data.items():
            if key in self.dict_sum:
                self.dict_sum[key] += value
                if self.cache:
                    self.dict_cache[key].append(value)
            else:
                self.dict_sum[key] = value
                if self.cache:
                    self.dict_cache[key] = [value]
        self.cnt += 1

    def average(self):
        data = {}
        for key, value in self.dict_sum.items():
            if not isinstance(value, str):
                value = value / self.cnt
                data[key] = value
        return data

    def reset(self):
        self.cnt = 0
        self.dict_sum = {}


def tbadd_dict(tbwriter, data_dict: dict, index, group="loss"):
    for key, value in data_dict.items():
        tbwriter.add_scalar(f"{group}/{key}", value, index)


def format_dict(dict_data: dict, ncols=None) -> str:
    output = ""
    ncols = ncols or 4096
    cnt = 0
    for key, value in dict_data.items():
        cnt = cnt + 1
        if isinstance(value, float):
            if abs(value) > 1e30:
                value = "inf"
            elif abs(value) > 1e-3:
                value = f"{value:.4f}".rstrip("0").rstrip(".")
            elif abs(value) > 1e-30:
                left, right = f"{value:.4e}".split("e")
                value = left.rstrip("0").rstrip(".") + "e" + right
            else:
                value = "0"
        output += f"{key}={value} | "
        if cnt == ncols:
            cnt = 0
            output += "\n"

    return output.strip("\n")


def pklsave(data, path):
    with open(path, "wb") as file:
        pkl.dump(data, file)


def pkload(path):
    with open(path, 'rb') as file:
        pkl_data = pkl.load(file)
    return pkl_data
