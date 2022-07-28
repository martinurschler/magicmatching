from typing import Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import numpy as np

import pytorch_lightning as pl

from utils.image_utils import read_rgb_image_to_single_channel_tensor, read_rgb_image_to_uint8_numpy_array
from utils.image_utils import normalize_2d_gray_image_for_nn
from utils.augmentation_utils import ImgAugTransformHomographicAdaptation

# not a nice import, goes one level up...
#from unet_model import MagicPointUNetModule


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, data_dir: str, train: bool) -> None:
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

        augmentation = False

        if augmentation == True:
            image = read_rgb_image_to_uint8_numpy_array(self.image_paths[index])
            augmentation = ImgAugTransformHomographicAdaptation()
            image = augmentation(image)
            image = normalize_2d_gray_image_for_nn(image)
        else:
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

    def predict_dataloader(self, shuffle=True):
        return DataLoader(self.coco_dataset_predict, num_workers=1, batch_size=1, shuffle=shuffle)


# class CocoWithHomAdaptDataset(torch.utils.data.Dataset):
#
#     def __init__(self, data_dir: str, absolute_path_of_magicpoint: str, train: bool) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.absolute_path_of_magicpoint = absolute_path_of_magicpoint
#
#         self.magicpoint = MagicPointUNetModule.load_from_checkpoint(self.absolute_path_of_magicpoint)
#         self.magicpoint.eval()
#
#         self.N_H = 3  # magic number from paper
#
#         # get files
#         which_data = "train" if train else "val"
#
#         base_path = Path(self.data_dir) / Path(which_data + '2014')
#         image_paths = list(base_path.iterdir())
#         self.image_paths = [str(p) for p in image_paths]
#         pass
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, index):
#         #print("index = ", index, self.image_paths[index])
#         image = read_rgb_image_to_uint8_numpy_array(self.image_paths[index])
#         image_torch_tensor = torch.Tensor(np.array([normalize_2d_gray_image_for_nn(image)]))
#
#         # apply homographic adaptation
#         accumulated_target = torch.Tensor(image_torch_tensor.shape)
#         with torch.no_grad():
#             # 1. just apply magic point to the input image
#             target_hat = self.magicpoint(image_torch_tensor)
#             accumulated_target += target_hat
#
#             # 2. do more predictions on augmented images, till we reach self.N_H
#             for homography_idx in range(self.N_H - 1):
#                 #print(homography_idx)
#                 # create a random homographic transformation
#                 augmentation = ImgAugTransformHomographicAdaptation()
#                 augmented_image = augmentation(image)
#                 augmented_image_torch_tensor = torch.Tensor(np.array([normalize_2d_gray_image_for_nn(augmented_image)]))
#                 target_hat = self.magicpoint(augmented_image_torch_tensor)
#                 accumulated_target += target_hat
#
#         # 3. average the accumulated response
#         accumulated_target /= float(self.N_H)
#
#         # convert to numpy tensor image
#         tensor_image = normalize_2d_gray_image_for_nn(image)
#
#         return tensor_image, accumulated_target
#
# # this class creates pairs of images and target labels for keypoint detection using homographic adaptation
# class CocoWithHomAdaptDataModule(pl.LightningDataModule):
#
#     def __init__(self, data_dir: str, absolute_path_of_magicpoint: str, batch_size: int = 4):
#         super().__init__()
#         self.data_dir = data_dir
#         self.absolute_path_of_magicpoint = absolute_path_of_magicpoint
#         self.batch_size = batch_size
#
#         self.coco_dataset_train = CocoWithHomAdaptDataset(self.data_dir, self.absolute_path_of_magicpoint, train=True)
#
#         pass
#
#     def train_dataloader(self):
#         return DataLoader(self.coco_dataset_train, num_workers=4, batch_size=self.batch_size, shuffle = True)