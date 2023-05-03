from imgaug import augmenters as iaa
import cv2
import numpy as np
from scipy.ndimage import measurements
"""
from Hover-Net augmentation
https://github.com/vqdang/hover_net/blob/master/dataloader/augs.py
"""

####
def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).

    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


####
def gaussian_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply Gaussian blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    ret = ret.astype(np.uint8)
    return [ret]


####
def median_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply median blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_hue(images, random_state, parents, hooks, range=(-8, 8)):
    """Perturbe the hue of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    hue = random_state.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_saturation(images, random_state, parents, hooks, range=(-0.2, 0.2)):
    """Perturbe the saturation of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = 1 + random_state.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, random_state, parents, hooks, range=(0.75, 1.25)):
    """Perturbe the contrast of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, random_state, parents, hooks, range=(-26, 26)):
    """Perturbe the brightness of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


def get_augmentation(mode, rng,input_shape=(256,256)):
    if mode == "train":
        print('Using train augmentation')
        shape_augs = [
            # * order = ``0`` -> ``cv2.INTER_NEAREST``
            # * order = ``1`` -> ``cv2.INTER_LINEAR``
            # * order = ``2`` -> ``cv2.INTER_CUBIC``
            # * order = ``3`` -> ``cv2.INTER_CUBIC``
            # * order = ``4`` -> ``cv2.INTER_CUBIC``
            # ! for pannuke v0, no rotation or translation, just flip to avoid mirror padding
            iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -A to +A percent (per axis)
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                shear=(-5, 5),  # shear by -5 to +5 degrees
                rotate=(-179, 179),  # rotate by -179 to +179 degrees
                order=0,  # default 0 use nearest neighbour
                backend="cv2",  # opencv for fast processing
                seed=rng,
            ),
            # set position to 'center' for center crop
            # else 'uniform' for random crop
            iaa.CropToFixedSize(
                input_shape[0], input_shape[1], position="center"
            ),
            iaa.Fliplr(0.5, seed=rng),
            iaa.Flipud(0.5, seed=rng),
        ]

        input_augs = [
            iaa.OneOf(
                [
                    iaa.Lambda(
                        seed=rng,
                        func_images= gaussian_blur,
                    ),
                    iaa.Lambda(
                        seed=rng,
                        func_images= median_blur,
                    ),
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                    ),
                ]
            ),
            iaa.Sequential(
                [
                    iaa.Lambda(
                        seed=rng,
                        func_images=add_to_hue,
                    ),
                    iaa.Lambda(
                        seed=rng,
                        func_images=add_to_saturation
                        ,
                    ),
                    iaa.Lambda(
                        seed=rng,
                        func_images=add_to_brightness
                        ,
                    ),
                    iaa.Lambda(
                        seed=rng,
                        func_images= add_to_contrast,
                    ),
                ],
                random_order=True,
            ),
        ]
    elif mode == "test":
        print('Using test augmentation')
        shape_augs = [
            # set position to 'center' for center crop
            # else 'uniform' for random crop
            iaa.CropToFixedSize(
                input_shape[0], input_shape[1], position="center"
            )
        ]
        input_augs = []

    return iaa.Sequential(shape_augs), iaa.Sequential(input_augs)