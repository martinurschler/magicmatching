import numpy as np
import cv2

from utils.nms_utils import getPtsFromHeatmap

def draw_interest_points(img, points, circle_radius = 1):
    """ Convert img in RGB and draw in green the interest points """
    img_rgb = np.stack([img, img, img], axis=2)
    for i in range(len(points)):
        cv2.circle(img_rgb, (points[i][0], points[i][1]), circle_radius, (0, 255, 0), -1)
    return img_rgb

def normalize_2d_gray_image_for_nn(image):
    '''Takes a uint8 tensor of H,W as input, produces a float tensor of 1,H,W between -1 and 1'''
    image = (image.astype('float32') - np.float32(127.0)) / np.float32(255.0)
    return np.array([image])

def unnormalize_2d_gray_image_for_opencv(image):
    '''Takes a float tensor of H,W as input, produces a uint8 tensor of H,W between 0 and 255'''
    image = np.array(image * np.float32(255.0) + np.float32(127.0), dtype=np.uint8)
    return image

def read_rgb_image_to_uint8_numpy_array(path, resize_lower = 384, resize_higher = 512):
    '''Shape of result image is [384, 512] or [512, 384]'''
    input_image = cv2.imread(path)
    if input_image.shape[0] < input_image.shape[1]:
        input_image = cv2.resize(input_image, (resize_higher, resize_lower),
                                 interpolation=cv2.INTER_AREA)
    else:
        input_image = cv2.resize(input_image, (resize_lower, resize_higher),
                                 interpolation=cv2.INTER_AREA)
    # H, W = input_image.shape[0], input_image.shape[1]
    # print("after resize, h, w", H, W)

    # convert RGB to gray (note: OpenCV reads channels as BGR not RGB
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return image

def read_rgb_image_to_single_channel_tensor(path, resize_lower = 384, resize_higher = 512):
    '''Shape of result image is [1, 384, 512] or [1, 512, 384]'''
    input_image = read_rgb_image_to_uint8_numpy_array(path, resize_lower, resize_higher)
    tensor_image = normalize_2d_gray_image_for_nn(input_image)
    return tensor_image

def convert_torch_tensor_to_numpy_2dimage(torch_tensor):
    # convert tensor img to numpy
    img_np = torch_tensor.numpy()

    # reshape to have single channel image suitable for opencv or point extraction
    h_dim = len(torch_tensor.shape) - 2
    w_dim = len(torch_tensor.shape) - 1

    img_2d = np.reshape(img_np, (img_np.shape[h_dim], img_np.shape[w_dim]))

    return img_2d

def get_groundtruth_keypoint_locations_from_heatmap(torch_tensor_heatmap):
    ''' input is an img as a torch tensor, which has height and width in its last two dimensions
         output is a list of keypoint locations consisting of (x,y) tuples
    '''
    heatmap = convert_torch_tensor_to_numpy_2dimage(torch_tensor_heatmap)

    rows = heatmap.shape[0]
    cols = heatmap.shape[1]
    keypoint_locations = []
    for y in range(rows):
        for x in range(cols):
            if heatmap[y, x] > 0.0:
                keypoint_locations.append((x, y))

    return keypoint_locations


def get_keypoint_locations_from_predicted_heatmap(torch_tensor_heatmap, nms_conf_thr = 0.5, nms_distance = 2):
    ''' input is an img as a torch tensor, which has height and width in its last two dimensions
        output is a list of keypoint locations consisting of (x,y) tuples
    '''
    #print(torch_tensor_heatmap.shape, len(torch_tensor_heatmap.shape))

    heatmap = convert_torch_tensor_to_numpy_2dimage(torch_tensor_heatmap)

    #print("heatmap min max", np.min(heatmap), np.max(heatmap))

    # clean up heatmap (clamp values below 0.0 or above 1.0)
    heatmap[heatmap < 0.0] = 0.0
    heatmap[heatmap > 1.0] = 1.0

    # scale responses such that the maximum is at 1, nms_conf_thr will be relative to that!
    heatmap /= np.max(heatmap)

    #print("heatmap min max scaled", np.min(heatmap), np.max(heatmap))

    predicted_keypoint_locations = getPtsFromHeatmap(heatmap, nms_conf_thr, nms_distance)
    #print("number of points from heatmap", predicted_keypoint_locations.shape[1])

    keypoints = []
    for pt_idx in range(predicted_keypoint_locations.shape[1]):
        keypoints.append(
            (int(predicted_keypoint_locations[0, pt_idx]), int(predicted_keypoint_locations[1, pt_idx])))

    return keypoints

def write_image_with_keypoints(torch_tensor_image, keypoints, filename):
    ''' input is an img as a torch tensor, which has height and width in its last two dimensions
        input is a list of keypoints as (x,y) tuples
    '''

    img_cv2 = convert_torch_tensor_to_numpy_2dimage(torch_tensor_image)
    img_8bit_cv2 = unnormalize_2d_gray_image_for_opencv(img_cv2)

    cv2.imwrite(filename, draw_interest_points(img_8bit_cv2, keypoints))

def generate_gt_heatmap_from_keypoint_list(keypoints, H, W):

    heatmap = np.zeros((H, W), dtype=np.float32)
    for kp in keypoints:
        heatmap[kp[1], kp[0]] = 1.0
    return heatmap