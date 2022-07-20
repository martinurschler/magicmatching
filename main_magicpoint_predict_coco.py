from pathlib import Path, PureWindowsPath

import numpy as np
import torch
import cv2

from unet_model import UNetModule
from datasets.coco_dataset import CocoDataModule
from utils.nms_utils import getPtsFromHeatmap
from utils.image_utils import draw_interest_points


if __name__ == "__main__":

    tmp_dir = "tmp"

    coco_base_path = PureWindowsPath("E:/01_Repos/xzho372/data/COCO")

    unet_magicpoint = UNetModule.load_from_checkpoint("current_best_model_overall.ckpt")

    coco = CocoDataModule(data_dir=coco_base_path)
    coco.setup()
    predict_loader = coco.predict_dataloader()

    count = 0
    unet_magicpoint.eval()
    for img in predict_loader:
        # make prediction
        with torch.no_grad():
            target_hat = unet_magicpoint(img)

            h_dim = 2
            w_dim = 3

            img_cv2 = img.numpy()
            img_cv2 = np.reshape(img_cv2, (img_cv2.shape[h_dim], img_cv2.shape[w_dim]))
            img_cv2 = img_cv2 * np.float32(255.0) + np.float32(127.0)

            target_hat_cv2 = target_hat.numpy()
            #print(target_hat_cv2.shape)
            target_hat_cv2 = np.reshape(target_hat_cv2, (target_hat_cv2.shape[h_dim], target_hat_cv2.shape[w_dim]))
            target_hat_cv2 = target_hat_cv2 * np.float32(255.0)
            #print("target hat min max", np.min(target_hat_cv2), np.max(target_hat_cv2))
            target_hat_cv2[target_hat_cv2 < 0.0] = 0.0
            target_hat_cv2[target_hat_cv2 > 255.0] = 255.0
            #print("target hat min max", np.min(target_hat_cv2), np.max(target_hat_cv2))

            prediction_locations = getPtsFromHeatmap(target_hat_cv2, 0.5 * 255.0, 1)
            print("number of points from heatmap", prediction_locations.shape[1])
            #print("prediction locations matrix:\n", prediction_locations)
            max_locations_target_hat = []
            for pt_idx in range(prediction_locations.shape[1]):
                max_locations_target_hat.append(
                    (int(prediction_locations[0, pt_idx]), int(prediction_locations[1, pt_idx])))

            print("writing output image")
            cv2.imwrite(str(Path(tmp_dir) / "predict_coco{}.png".format(count)),
                        draw_interest_points(np.array(img_cv2, dtype=np.uint8), max_locations_target_hat))

            count += 1

            if count > 5:
                break