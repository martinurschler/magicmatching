from pathlib import Path, PureWindowsPath

import torch
import cv2
import numpy as np

from unet_model import MagicPointUNetModule
from utils.image_utils import normalize_2d_gray_image_for_nn, get_keypoint_locations_from_predicted_heatmap, draw_interest_points
from utils.image_utils import generate_gt_heatmap_from_keypoint_list


#def intersection(lst1, lst2):
#    # Use of hybrid method
#    temp = set(lst2)
#    lst3 = [value for value in lst1 if value in temp]
#    return lst3

def merge_keypoints(keypoints1, keypoints2, distance_threshold_px = 2):

    merged_keypoints = []

    pts2 = keypoints2.copy()
    for kp1 in keypoints1:
        merged_keypoints.append(kp1)
        pts2_ignore_indices = []
        for idx, kp2 in enumerate(pts2):
            if np.abs(kp1[0]-kp2[0]) <= distance_threshold_px and np.abs(kp1[1]-kp2[1]) <= distance_threshold_px:
                pts2_ignore_indices.append(idx)
        if len(pts2_ignore_indices) >= 0:
            tmp = []
            for idx, kp2 in enumerate(pts2):
                if idx not in pts2_ignore_indices:
                    tmp.append(kp2)
            pts2 = tmp

    merged_keypoints += pts2

    return merged_keypoints

if __name__ == "__main__":

    DEBUG_OUTPUT = False

    N_H = 100
    RESAMPLE_SIZE_SQUARE = 384
    NMS_CONF_THRESHOLD = 0.15 # relative to a 0 - 1 range

    HA_MODE_AVERAGING = True # if this is False, we use a maximum mode to aggregate heatmaps

    tmp_dir = "tmp"
    if not Path(tmp_dir).is_dir():
        try:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: {e.strerror}")
            exit()

    coco_base_path = PureWindowsPath("E:/01_Repos/xzho372/data/COCO")
    which_coco = 'train2014'

    model_path = str(Path("data") / "pretrained_archive" / "current_best_magicpoint_model_v1.ckpt")
    unet_magicpoint = MagicPointUNetModule.load_from_checkpoint(model_path)
    unet_magicpoint.eval()

    base_path = Path(coco_base_path) / Path(which_coco)
    coco_image_paths = [str(p) for p in list(base_path.iterdir())]
    coco_image_names = [p.stem for p in list(base_path.iterdir())]
    print(len(coco_image_paths))

    output_path = Path(tmp_dir) / Path(f"coco_{which_coco}_ha")

    with torch.no_grad():
        for img_idx in range(len(coco_image_paths)):

            print(f"processing image {img_idx}")

            input_image = cv2.imread(coco_image_paths[img_idx])
            # convert RGB to gray (note: OpenCV reads channels as BGR not RGB
            gray_image = cv2.resize(cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY),
                                    (RESAMPLE_SIZE_SQUARE, RESAMPLE_SIZE_SQUARE),
                                    interpolation=cv2.INTER_AREA)

            img_name = coco_image_names[img_idx]
            filename = str(output_path / Path(f"{img_name}_ha.png"))
            cv2.imwrite(filename, gray_image)

            if DEBUG_OUTPUT:
                filename = str(Path(tmp_dir) / f"test_ha_{img_idx}.png")
                cv2.imwrite(filename, gray_image)

            img_width = RESAMPLE_SIZE_SQUARE
            img_height = RESAMPLE_SIZE_SQUARE

            # apply the magicpoint predictor to not warped image
            tensor_image = np.array([normalize_2d_gray_image_for_nn(gray_image)])
            # initialize the accumulated t hat image
            t_hat = unet_magicpoint(torch.Tensor(tensor_image))
            #print(t_hat.shape)
            t_hat_identity = t_hat.numpy().reshape(t_hat.shape[len(t_hat.shape)-2], t_hat.shape[len(t_hat.shape)-1])

            #print("t hat identity min and max", np.min(t_hat_identity), np.max(t_hat_identity))
            t_hat_identity[t_hat_identity < 0.0] = 0.0
            t_hat_identity[t_hat_identity > 1.0] = 1.0
            t_hat_identity /= np.max(t_hat_identity)
            #print("t hat identity min and max", np.min(t_hat_identity), np.max(t_hat_identity))

            #filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_target_0identity.png")
            #cv2.imwrite(filename, np.float32(255.0) * t_hat_identity)

            predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(torch.Tensor(t_hat_identity),
                                                                                nms_conf_thr=NMS_CONF_THRESHOLD)
            if DEBUG_OUTPUT:
                filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_ha_predictions_0identity.png")
                cv2.imwrite(filename, draw_interest_points(gray_image, predicted_keypoints))

            t_hat_accumulated = np.zeros(t_hat_identity.shape, dtype=np.float32)
            # counts for each pixel how often it leads to a valid unwarped heatmap pixel, init with 1 due to identity!
            mask_for_averaging_heatmaps = np.ones(t_hat_identity.shape, dtype=np.uint32)

            # apply a number N_H of homographic adaptations ha
            for ha_idx in range(N_H):

                # compute the perspective transform matrix and then apply it
                p0 = (np.abs(np.random.normal(0.0, 0.125 * img_width)),
                      np.abs(np.random.normal(0.0, 0.125 * img_height)))
                p1 = (np.abs(np.random.normal(0.0, 0.125 * img_width)),
                      np.abs(np.random.normal(0.0, 0.125 * img_height)))
                p2 = (np.abs(np.random.normal(0.0, 0.125 * img_width)),
                      np.abs(np.random.normal(0.0, 0.125 * img_height)))
                p3 = (np.abs(np.random.normal(0.0, 0.125 * img_width)),
                      np.abs(np.random.normal(0.0, 0.125 * img_height)))

                rect = np.array([
                                 [p0[0], p0[1]],
                                 [img_width - 1 - p1[0], p1[1]],
                                 [img_width - 1 - p2[0], img_height - 1 - p2[1]],
                                 [p3[0], img_height - 1 - p3[1]]], dtype="float32")
                dst = np.array([
                                [0, 0],
                                [RESAMPLE_SIZE_SQUARE - 1, 0],
                                [RESAMPLE_SIZE_SQUARE - 1, RESAMPLE_SIZE_SQUARE - 1],
                                [0, RESAMPLE_SIZE_SQUARE - 1]], dtype="float32")

                homography = cv2.getPerspectiveTransform(rect, dst)
                #print("homography", M)
                warped_image = cv2.warpPerspective(gray_image, homography, (RESAMPLE_SIZE_SQUARE, RESAMPLE_SIZE_SQUARE))

                #filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_{ha_idx}_warped.png")
                #cv2.imwrite(filename, warped_image)

                # apply the magicpoint predictor
                tensor_image = np.array([normalize_2d_gray_image_for_nn(warped_image)])

                t_hat = unet_magicpoint(torch.Tensor(tensor_image))

                # undo the warping
                unwarped_target = cv2.warpPerspective(t_hat.numpy().reshape(t_hat.shape[len(t_hat.shape)-2], t_hat.shape[len(t_hat.shape)-1]),
                                                      np.linalg.inv(homography),
                                                      (RESAMPLE_SIZE_SQUARE, RESAMPLE_SIZE_SQUARE))

                if HA_MODE_AVERAGING == True:
                    unwarped_target[unwarped_target < 0.0] = 0.0
                    unwarped_target[unwarped_target > 1.0] = 1.0
                    t_hat_accumulated += unwarped_target
                    #t_hat_accumulated[t_hat_accumulated > 1.0] = 1.0 # saturate to be more robust in averaging
                else:
                    t_hat_accumulated = np.maximum(t_hat_accumulated, unwarped_target)
                    pass

                unwarped_image = cv2.warpPerspective(warped_image, np.linalg.inv(homography),
                                                     (RESAMPLE_SIZE_SQUARE, RESAMPLE_SIZE_SQUARE))

                # analyze the unwarped image
                _, thresholded_image = cv2.threshold(unwarped_image, 0, 1, cv2.THRESH_BINARY)
                mask_for_averaging_heatmaps += thresholded_image

                #filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_{ha_idx}_warped_unwarped.png")
                #cv2.imwrite(filename, unwarped_image)

                predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(torch.Tensor(unwarped_target),
                                                                                    nms_conf_thr=NMS_CONF_THRESHOLD)
                #filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_{ha_idx}_ha_warped_unwarped_predictions.png")
                #cv2.imwrite(filename, draw_interest_points(unwarped_image, predicted_keypoints))

                #filename = str(Path("tmp") / f"test_ha_{img_idx}_{ha_idx}_target.png")
                #cv2.imwrite(filename, t_hat)

            #print("mask min max", np.min(mask_for_averaging_heatmaps), np.max(mask_for_averaging_heatmaps))

            if HA_MODE_AVERAGING == True:
                t_hat_accumulated += t_hat_identity
                t_hat_accumulated = np.true_divide(t_hat_accumulated, mask_for_averaging_heatmaps.astype(np.float32))

            # finally, add the original identity transformed version by max operation
            #t_hat_accumulated = np.maximum(t_hat_accumulated, t_hat_identity)
            # t_hat_accumulated[t_hat_accumulated > 1.0] = 1.0

            #print("t hat accumulated min and max", np.min(t_hat_accumulated), np.max(t_hat_accumulated))
            t_hat_accumulated[t_hat_accumulated < 0.0] = 0.0
            t_hat_accumulated[t_hat_accumulated > 1.0] = 1.0
            t_hat_accumulated /= np.max(t_hat_accumulated)
            #print("t hat accumulated min and max", np.min(t_hat_accumulated), np.max(t_hat_accumulated))

            #filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_target_1accumulated.png")
            #cv2.imwrite(filename, np.float32(255.0) * t_hat_accumulated)


            predicted_keypoints1 = get_keypoint_locations_from_predicted_heatmap(torch.Tensor(t_hat_identity),
                                                                                 nms_conf_thr=NMS_CONF_THRESHOLD)
            predicted_keypoints2 = get_keypoint_locations_from_predicted_heatmap(torch.Tensor(t_hat_accumulated),
                                                                                 nms_conf_thr=NMS_CONF_THRESHOLD)
            predicted_keypoints = merge_keypoints(predicted_keypoints1, predicted_keypoints2, distance_threshold_px=2)

            if DEBUG_OUTPUT:
                filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_ha_predictions_1accumulated.png")
                cv2.imwrite(filename, draw_interest_points(gray_image, predicted_keypoints))

            gt_heatmap = generate_gt_heatmap_from_keypoint_list(predicted_keypoints, RESAMPLE_SIZE_SQUARE, RESAMPLE_SIZE_SQUARE)

            #filename = str(Path(tmp_dir) / f"test_ha_{img_idx}_ha_predictions_2final_heatmap.png")
            img_name = coco_image_names[img_idx]
            filename = str(output_path / Path(f"{img_name}_ha_target.png"))
            cv2.imwrite(filename, np.float32(255.0) * gt_heatmap)

            #if img_idx >= 3:
            #    break