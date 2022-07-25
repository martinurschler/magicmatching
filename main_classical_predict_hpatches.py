from pathlib import Path, PureWindowsPath
import os

import numpy as np
import cv2
import torch

from datasets.hpatches_dataset import HPatchesDataModule
from utils.image_utils import get_keypoint_locations_from_predicted_heatmap, write_image_with_keypoints
from utils.image_utils import unnormalize_2d_gray_image_for_opencv, convert_torch_tensor_to_numpy_2dimage

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

    hpatches = HPatchesDataModule(data_root_dir=DATA_ROOT_PATH)
    hpatches.prepare_data(download = True)
    hpatches.setup()
    predict_loader = hpatches.predict_dataloader()

    count = 0
    for img in predict_loader:

        do_harris = False
        do_fast = False
        do_shi = True

        img_8bit = unnormalize_2d_gray_image_for_opencv(convert_torch_tensor_to_numpy_2dimage(img))

        if do_harris:
            target_hat_harris = cv2.cornerHarris(img_8bit, 4, 3, 0.04).astype(np.float32)
            #print("target hat min max", np.min(target_hat_harris), np.max(target_hat_harris))
            target_hat_harris[target_hat_harris < 0.0] = 0.0
            target_hat_harris /= np.max(target_hat_harris)
            #print("target hat min max", np.min(target_hat_harris), np.max(target_hat_harris))
            target_hat_harris = torch.Tensor(target_hat_harris)

            predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(target_hat_harris, nms_conf_thr=0.2)
            filename = str(Path(tmp_dir) / "predict_hpatches{}_harris.png".format(count))
            write_image_with_keypoints(img, predicted_keypoints, filename)

        if do_fast:
            fast_detector = cv2.FastFeatureDetector_create(10)
            fast_corners = fast_detector.detect(img_8bit, mask=None)
            target_hat_fast = np.zeros(img_8bit.shape, np.float32)
            for c in fast_corners:
                target_hat_fast[tuple(np.flip(np.int0(c.pt), 0))] = c.response
            #print("target hat min max", np.min(target_hat_fast), np.max(target_hat_fast))
            target_hat_fast /= np.max(target_hat_fast)
            target_hat_fast = torch.Tensor(target_hat_fast)

            predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(target_hat_fast)
            filename = str(Path(tmp_dir) / "predict_hpatches{}_fast.png".format(count))
            write_image_with_keypoints(img, predicted_keypoints, filename)

        if do_shi:
            target_hat_shi = np.zeros(img_8bit.shape, np.float32)
            thresholds = np.linspace(0.0001, 1, 300, endpoint=False)
            #print(thresholds)
            for t in thresholds:
                corners = cv2.goodFeaturesToTrack(img_8bit, 300, t, 5)
                if corners is not None:
                    corners = corners.astype(np.int32)
                    target_hat_shi[(corners[:, 0, 1], corners[:, 0, 0])] = t
            #print("target hat min max", np.min(target_hat_shi), np.max(target_hat_shi))
            target_hat_shi /= np.max(target_hat_shi)
            target_hat_shi = torch.Tensor(target_hat_shi)

            predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(target_hat_shi)
            filename = str(Path(tmp_dir) / "predict_hpatches{}_shi.png".format(count))
            write_image_with_keypoints(img, predicted_keypoints, filename)

        count += 1

        if count > 20:
            break