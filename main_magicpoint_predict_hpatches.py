from pathlib import Path, PureWindowsPath
import os

import torch

from unet_model import MagicPointUNetModule
from datasets.hpatches_dataset import HPatchesDataModule
from utils.image_utils import get_keypoint_locations_from_predicted_heatmap, write_image_with_keypoints

if __name__ == "__main__":

    #pl.seed_everything(42, workers=True)

    DATA_ROOT_PATH = str(Path(os.getcwd()) / "data" / "HPatches")

    tmp_dir = "tmp"
    if not Path(tmp_dir).is_dir():
        try:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: {e.strerror}")
            exit()

    model_path = str(Path("data") / "pretrained_archive" / "current_best_magicpoint_model_v1.ckpt")
    unet_magicpoint = MagicPointUNetModule.load_from_checkpoint(model_path)

    hpatches = HPatchesDataModule(data_root_dir=DATA_ROOT_PATH)
    hpatches.prepare_data(download = True)
    hpatches.setup()
    predict_loader = hpatches.predict_dataloader()

    count = 0
    unet_magicpoint.eval()
    for img in predict_loader:
        # make prediction
        with torch.no_grad():
            target_hat = unet_magicpoint(img)

            predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(target_hat)
            filename = str(Path(tmp_dir) / "predict_hpatches{}.png".format(count))
            write_image_with_keypoints(img, predicted_keypoints, filename)

            count += 1

            if count > 20:
                break