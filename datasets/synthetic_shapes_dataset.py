import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from datasets.synthetic_shapes_draw_utils import generate_background, draw_star, draw_checkerboard, draw_polygon, draw_cube
from datasets.synthetic_shapes_draw_utils import draw_stripes, draw_ellipses, draw_multiple_polygons, draw_lines, draw_gaussian_noise
from utils.image_utils import normalize_2d_gray_image_for_nn
from utils.augmentation_utils import ImgAugTransform

class SyntheticShapesDataset(torch.utils.data.Dataset):

    def __init__(self, image_size, samples_per_epoch, apply_augmentation: bool = False):
        super().__init__()
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.apply_augmentation = apply_augmentation
        pass

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        """
        :param index:
        :return:
            labels_2D: tensor(1, H, W)
            image: tensor(1, H, W)
        """

        img = generate_background()
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)

        # these are the potential drawing operations
        drawing_operations = [draw_star, draw_ellipses, draw_checkerboard,
                              draw_lines, draw_polygon, draw_cube,
                              draw_stripes, draw_multiple_polygons, draw_gaussian_noise]

        # call a randomly selected drawing operation
        random_index = np.random.randint(0, len(drawing_operations))
        points = drawing_operations[random_index](img)

        if self.apply_augmentation:
            augmentation = ImgAugTransform()
            img = augmentation(img)

        # prepare the target function
        target = np.zeros(self.image_size, dtype=np.float32)
        # set target signal to 1 for point locations
        for pt_idx in range(points.shape[0]):
            x1 = points[pt_idx, 0]
            y1 = points[pt_idx, 1]
            target[y1, x1] = 1.0

        target = np.array([target])

        # bring images into range -1 ... 1 (approximately)
        normalized_img = normalize_2d_gray_image_for_nn(img)

        return normalized_img, target

class SyntheticShapesDataModule(pl.LightningDataModule):

    def __init__(self, batches_per_epoch, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.samples_per_epoch = batches_per_epoch * self.batch_size

        shapes_image_size = (192, 256)

        self.stateless_train_dataset = SyntheticShapesDataset(image_size=shapes_image_size,
                                                              samples_per_epoch=self.samples_per_epoch,
                                                              apply_augmentation=True)
        self.stateless_val_dataset = SyntheticShapesDataset(image_size=shapes_image_size,
                                                            samples_per_epoch=100 * self.batch_size)
        self.stateless_test_dataset = SyntheticShapesDataset(image_size=shapes_image_size,
                                                             samples_per_epoch=100 * self.batch_size)

        self.stateless_predict_dataset = SyntheticShapesDataset(image_size=shapes_image_size,
                                                                samples_per_epoch=1, apply_augmentation=False)

        #self.train_specific_transform = transforms.Compose([transforms.ToTensor()])
        #self.general_transform = transforms.Compose([transforms.ToTensor()])

        pass

    #def prepare_data(self) -> None:
    #    '''
    #    Used for downloading and tokenizing data, essentially one step prep before multi threaded processing starts.
    #    We could use it to check if fixed number of synthetic shapes were already generated on disk, and if so, load them
    #    '''
    #    pass

    #def setup(self, stage: Optional[str] = None) -> None:
    #    '''
    #    '''
    #    pass

    def train_dataloader(self):
        return DataLoader(self.stateless_train_dataset, num_workers=4, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.stateless_val_dataset, num_workers=4, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.stateless_test_dataset, num_workers=4, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.stateless_predict_dataset, num_workers=1, batch_size=1, shuffle=False)