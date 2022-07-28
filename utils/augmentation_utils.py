import numpy as np
from imgaug import augmenters as iaa

class ImgAugTransformHomographicAdaptation:

    def __init__(self):

        # homographic adaptation consists of: cropping at the center, scaling, translating, rotating and
        # symmetric perspective distortion

        # let's try the perspective transform from iaa first, it seems to be a perfect fit!
        self.aug = iaa.PerspectiveTransform(scale=(0.01, 0.20), keep_size=True)

        pass

    def __call__(self, img):
        '''
        :param img: an 8bit uint image passed in as a numpy array
        :return: an augmented 8bit uint image passed in as a numpy array
        '''
        img = self.aug.augment_image(img)
        return img

class ImgAugTransformPhotometric:

    def __init__(self):

        # 50% chance to also apply a gaussian blur with 5x5 kernel
        # do_blur = np.random.randint(0, 2)
        # if do_blur == 1:
        #    img = cv2.GaussianBlur(img, (5, 5), 0)

        self.aug = iaa.Sequential([iaa.Identity()])

        # There is a 2/3 chance to actually apply the augmentation pipeline, 1/3 chance to do nothing
        actually_do_augmentations = np.random.randint(0, 3)
        if actually_do_augmentations == 0:
            return

        do_random_brightness = True
        random_brightness_max_abs_offset = 75
        do_random_contrast = True
        random_contrast_strength_range = [0.3, 1.8]
        do_additive_gaussian_noise = True
        additive_gaussian_noise_stddev_range = [0, 10]
        do_impulse_noise = True
        impulse_noise_prob_range = [0, 0.0035]
        do_gaussian_blur = True
        gaussian_blur_sigma = 0.2

        all_augmentations = []

        # random brightness
        if do_random_brightness:
            aug = iaa.Add((-random_brightness_max_abs_offset, random_brightness_max_abs_offset))
            all_augmentations.append(aug)

        # random contrast
        if do_random_contrast:
            aug = iaa.LinearContrast((random_contrast_strength_range[0], random_contrast_strength_range[1]))
            all_augmentations.append(aug)

        # additive Gaussian noise
        if do_additive_gaussian_noise:
            aug = iaa.AdditiveGaussianNoise((additive_gaussian_noise_stddev_range[0], additive_gaussian_noise_stddev_range[1]))
            all_augmentations.append(aug)

        # impulse noise
        if do_impulse_noise:
            aug = iaa.ImpulseNoise(p=(impulse_noise_prob_range[0], impulse_noise_prob_range[1]))
            all_augmentations.append(aug)

        # Gaussian blur
        if do_gaussian_blur:
            aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(gaussian_blur_sigma)))
            all_augmentations.append(aug)

        self.aug = iaa.Sequential(all_augmentations, random_order=True)

        pass

    def __call__(self, img):
        '''
        :param img: an 8bit uint image passed in as a numpy array
        :return: an augmented 8bit uint image passed in as a numpy array
        '''
        img = self.aug.augment_image(img)
        return img