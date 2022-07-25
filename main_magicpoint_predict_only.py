from pathlib import Path

import torch

from unet_model import MagicPointUNetModule
from datasets.synthetic_shapes_dataset import SyntheticShapesDataModule

from utils.image_utils import get_keypoint_locations_from_predicted_heatmap, write_image_with_keypoints
from utils.image_utils import get_groundtruth_keypoint_locations_from_heatmap
from utils.evaluation_utils import computeDetectionPerformanceAndLocalizationError


if __name__ == "__main__":

    #pl.seed_everything(42, workers=True)

    BATCHES_PER_EPOCH = 1000
    BATCH_SIZE = 32
    tmp_dir = "tmp"
    if not Path(tmp_dir).is_dir():
        try:
            Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: {e.strerror}")
            exit()

    syntheticshapes = SyntheticShapesDataModule(BATCHES_PER_EPOCH, BATCH_SIZE)

    model_path = str(Path("data") / "pretrained_archive" / "current_best_magicpoint_model_v1.ckpt")
    unet_magicpoint = MagicPointUNetModule.load_from_checkpoint(model_path)

    predict_loader = syntheticshapes.predict_dataloader()
    count = 0
    unet_magicpoint.eval()
    for batch in predict_loader:
        img, target = batch

        # make prediction
        with torch.no_grad():
            target_hat = unet_magicpoint(img)

            predicted_keypoints = get_keypoint_locations_from_predicted_heatmap(target_hat)
            filename = str(Path(tmp_dir) / "predict_img_ovly_hat{}.png".format(count))
            write_image_with_keypoints(img, predicted_keypoints, filename)

            groundtruth_keypoints = get_groundtruth_keypoint_locations_from_heatmap(target)
            filename = str(Path(tmp_dir) / "predict_img_ovly_gt{}.png".format(count))
            write_image_with_keypoints(img, groundtruth_keypoints, filename)

            # print(img.shape, img.dtype)
            # print(target.shape, target.dtype)
            # print(target_hat.shape, target_hat.dtype)
            #
            # h_dim = 2
            # w_dim = 3
            #
            # img_cv2 = img.numpy()
            # img_cv2 = np.reshape(img_cv2, (img_cv2.shape[h_dim], img_cv2.shape[w_dim]))
            # img_cv2 = img_cv2 * np.float32(255.0) + np.float32(127.0)
            #
            # target_cv2 = target.numpy()
            # target_cv2 = np.reshape(target_cv2, (target_cv2.shape[h_dim], target_cv2.shape[w_dim]))
            # target_cv2 = target_cv2 * np.float32(255.0)
            #
            # target_hat_cv2 = target_hat.numpy()
            # target_hat_cv2 = np.reshape(target_hat_cv2, (target_hat_cv2.shape[h_dim], target_hat_cv2.shape[w_dim]))
            # target_hat_cv2 = target_hat_cv2 * np.float32(255.0)
            # print("target hat min max", np.min(target_hat_cv2), np.max(target_hat_cv2))
            # target_hat_cv2[target_hat_cv2 < 0.0] = 0.0
            # target_hat_cv2[target_hat_cv2 > 255.0] = 255.0
            # print("target hat min max", np.min(target_hat_cv2), np.max(target_hat_cv2))
            #
            # final_loss = np.mean((target_hat_cv2 - target_cv2) ** 2)
            # print(f"mean squared loss of uint8 target and prediction {final_loss}")
            #
            # #print(target_cv2.shape, target_cv2.shape[0])
            # rows = target_cv2.shape[0]
            # cols = target_cv2.shape[1]
            # max_locations_target = []
            # for y in range(rows):
            #     for x in range(cols):
            #         if target_cv2[y, x] == 255.0:
            #             #print("col", x, "row", y)
            #             max_locations_target.append((x, y))
            # max_locations_target = sorted(max_locations_target)
            #
            # prediction_locations = getPtsFromHeatmap(target_hat_cv2, 0.5 * 255.0, 1)
            # print("number of points from heatmap", prediction_locations.shape[1])
            # print("prediction locations matrix:\n", prediction_locations)
            # max_locations_target_hat = []
            # for pt_idx in range(prediction_locations.shape[1]):
            #     max_locations_target_hat.append((int(prediction_locations[0, pt_idx]), int(prediction_locations[1, pt_idx])))
            # max_locations_target_hat = sorted(max_locations_target_hat)
            #
            # cv2.imwrite(str(Path(tmp_dir) / "predict_img{}.png".format(count)), np.array(img_cv2, dtype=np.uint8))
            # cv2.imwrite(str(Path(tmp_dir) / "predict_target{}.png".format(count)), np.array(target_cv2, dtype=np.uint8))
            # cv2.imwrite(str(Path(tmp_dir) / "predict_target_hat{}.png".format(count)),
            #             np.array(target_hat_cv2, dtype=np.uint8))
            #
            # print("points to write gt", max_locations_target)
            # cv2.imwrite(str(Path(tmp_dir) / "predict_img_ovly_gt{}.png".format(count)),
            #             draw_interest_points(np.array(img_cv2, dtype=np.uint8), max_locations_target))
            # print("points to write pred", max_locations_target_hat)
            # cv2.imwrite(str(Path(tmp_dir) / "predict_img_ovly_hat{}.png".format(count)),
            #             draw_interest_points(np.array(img_cv2, dtype=np.uint8), max_locations_target_hat))

            computeDetectionPerformanceAndLocalizationError(groundtruth_keypoints, predicted_keypoints)

            count += 1


