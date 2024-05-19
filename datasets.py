import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import glob
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImageDict,
    RandRotateDict,
    RandRotate90Dict,
    RandFlipDict,
    RandSpatialCropDict,
    ScaleIntensityRangeDict,
)


class All_As_Tensor(MapTransform):
    def __init__(self, keys=None, dtype=torch.float, device="cpu"):
        self.keys = keys
        self.dtype = dtype
        self.device = device

    def __call__(self, data):
        d = dict(data)
        key_list = self.keys or d.keys()
        for key in key_list:
            tmp = d[key]
            if isinstance(tmp, torch.Tensor):
                tmp = tmp.numpy()
                tmp = torch.tensor(tmp, dtype=self.dtype, device=self.device)
                d[key] = tmp
        return d


class SIDDDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        root = "./datasets/SIDD/"

        train_HQ = sorted(glob.glob(f"{root}/train/*/*GT*.png"))
        train_dict = [{"name": os.path.basename(HQ), "HQ": HQ, "LQ": HQ.replace("GT", "noisy")} for HQ in train_HQ]

        val_HQ = sorted(glob.glob(f"{root}/val/*/*GT*.png"))
        val_dict = [{"name": os.path.basename(HQ), "HQ": HQ, "LQ": HQ.replace("GT", "noisy")} for HQ in val_HQ]

        self.files = None
        self.transforms = None

        self.file_dict = dict(train=train_dict, val=val_dict)
        keys = ['HQ', 'LQ']
        self.transforms_dict = {
            "train": Compose(
                [
                    LoadImageDict(keys, ensure_channel_first=True, reverse_indexing=False),
                    ScaleIntensityRangeDict(keys, a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                    RandFlipDict(keys, prob=0.5, spatial_axis=1),
                    RandRotate90Dict(keys, prob=0.2),
                    RandSpatialCropDict(keys, roi_size=[128, 128]),
                    All_As_Tensor(),
                ]
            ),
            "val": Compose(
                [
                    LoadImageDict(keys, ensure_channel_first=True, reverse_indexing=False),
                    ScaleIntensityRangeDict(keys, a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
                    All_As_Tensor(),
                ]
            ),
        }
        self.select('train')

    def select(self, state='train'):
        self.state = state
        self.files = self.file_dict[state]
        self.transforms = self.transforms_dict[state]
        return self

    def __getitem__(self, idx):
        item_dict = self.files[idx]
        item = self.transforms(item_dict)

        return item

    def __len__(self):
        return len(self.files)


class GaussianNoiseDataset(Dataset):

    def __init__(self, root_dirs, std=50, color=True, train=True):
        super(GaussianNoiseDataset, self).__init__()

        self.std = std / 255.0
        self.color = "RGB" if color else "L"

        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.image_paths = []
        for cur_path in root_dirs:
            self.image_paths += [
                os.path.join(cur_path, file).replace('\\', '/') for file in os.listdir(cur_path) if file.endswith(('png', 'jpg', 'jpeg', 'bmp'))
            ]
        self.image_paths.sort()

        if train:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(128),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        name = os.path.basename(path)
        HQ = Image.open(path).convert(self.color)
        HQ = self.transforms(HQ)
        LQ = torch.randn(HQ.shape) * self.std + HQ
        item = {"name": name, "HQ": HQ, "LQ": LQ}
        return item

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset = SIDDDataset().select("train")
    dataset = GaussianNoiseDataset(["./datasets/CBSD68"], color='GRAY', train=False)
    from monai.data import DataLoader

    loader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
    print(len(dataset))
    for x in loader:
        # x = dataset[i]
        HQ, LQ = x['HQ'], x['LQ']

        break
