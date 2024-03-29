import glob
import os.path as osp

import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.data_utils import make_dataset


class ImagesDataset(Dataset):
    def __init__(self, source_root, latent_root, source_transform=None):
        self.source_paths = sorted(make_dataset(source_root))
        self.latent_paths = sorted(glob.glob(osp.join(latent_root, "*.pt")))
        self.source_transform = source_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert("RGB")

        latent = torch.load(self.latent_paths[index])

        if self.source_transform:
            from_im = self.source_transform(from_im)

        return fname, from_im, latent
