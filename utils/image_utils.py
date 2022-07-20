import numpy as np
import cv2

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

def read_rgb_image_to_single_channel_tensor(path, resize_lower = 384, resize_higher = 512):
    '''Shape of result image is [1, 384, 512] or [1, 512, 384]'''
    input_image = cv2.imread(path)
    # print(f"path: {path}, image: {input_image.shape}")
    if input_image.shape[0] < input_image.shape[1]:
        input_image = cv2.resize(input_image, (resize_higher, resize_lower),
                                 interpolation=cv2.INTER_AREA)
    else:
        input_image = cv2.resize(input_image, (resize_lower, resize_higher),
                                 interpolation=cv2.INTER_AREA)
    # H, W = input_image.shape[0], input_image.shape[1]
    # print("after resize, h, w", H, W)

    # convert RGB to gray (note: OpenCV reads channels as BGR not RGB
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    tensor_image = normalize_2d_gray_image_for_nn(input_image)
    return tensor_image