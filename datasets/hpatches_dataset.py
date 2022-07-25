from typing import Optional
from pathlib import Path
import os
from urllib.error import URLError

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_and_extract_archive, check_integrity

import pytorch_lightning as pl

from utils.image_utils import read_rgb_image_to_single_channel_tensor

class HPatchesDataset(torch.utils.data.Dataset):

    def __init__(self, data_raw_dir: str, train: bool):
        super().__init__()
        self.data_unpacked_dir = Path(data_raw_dir) / "hpatches-sequences-release"
        print("data is in folder:", self.data_unpacked_dir)

        # get files
        which_data = "train" if train else "val"

        self.image_paths = []

        scene_paths = list(self.data_unpacked_dir.iterdir())
        #print(scene_paths)
        for scene_path in scene_paths:
            per_scene_filename_paths = list(scene_path.iterdir())
            #print(per_scene_filename_paths)
            for image_path in per_scene_filename_paths:
                if image_path.suffix == ".ppm":
                    self.image_paths.append(str(image_path))
        #print(self.image_paths)
        pass

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        #print("index = ", index, self.image_paths[index])
        image = read_rgb_image_to_single_channel_tensor(self.image_paths[index])
        return image



class HPatchesDataModule(pl.LightningDataModule):

    download_location_url = [
        "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/"
    ]

    resource_files = [
        "hpatches-sequences-release.tar.gz"
    ]

    location_of_downloaded_zip_file = "raw"

    def _check_exists(self) -> bool:
        return all(
            check_integrity(str(Path(self.data_raw_dir) / url))
            for url in self.resource_files
        )

    def __init__(self, data_root_dir: str, batch_size: int = 4) -> None:
        super().__init__()
        self.data_root_dir = data_root_dir
        self.data_raw_dir = str(Path(self.data_root_dir) / self.location_of_downloaded_zip_file)
        self.batch_size = batch_size

        pass

    def prepare_data(self, download: bool = False) -> None:

        if download:
            self.download_dataset()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("predict", None):
            self.hpatches_dataset_predict = HPatchesDataset(self.data_raw_dir, train=False)
        pass

    def predict_dataloader(self):
        return DataLoader(self.hpatches_dataset_predict, num_workers=1, batch_size=1, shuffle=False)


    def download_dataset(self) -> None:
        """Download the HPatches data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.data_root_dir, exist_ok=True)
        os.makedirs(self.data_raw_dir, exist_ok=True)

        # download files
        for filename in self.resource_files:
            for download_location in self.download_location_url:
                url = f"{download_location}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_and_extract_archive(url, download_root=self.data_raw_dir, filename=filename)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")