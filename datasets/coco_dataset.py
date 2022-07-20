from typing import Optional
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from utils.image_utils import read_rgb_image_to_single_channel_tensor

class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: str, train: bool):
        super().__init__()
        self.data_dir = data_dir

        # get files
        which_data = "train" if train else "val"

        base_path = Path(self.data_dir) / Path(which_data + '2014')
        image_paths = list(base_path.iterdir())
        names = [p.stem for p in image_paths]
        self.image_paths = [str(p) for p in image_paths]
        #files = {'image_paths': image_paths, 'names': names}
        print(len(self.image_paths))
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        #print("index = ", index, self.image_paths[index])
        image = read_rgb_image_to_single_channel_tensor(self.image_paths[index])
        return image

class CocoDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("predict", None):
            self.coco_dataset_predict = CocoDataset(self.data_dir, train=False)
        pass

    def predict_dataloader(self):
        return DataLoader(self.coco_dataset_predict, num_workers=1, batch_size=1, shuffle=True)