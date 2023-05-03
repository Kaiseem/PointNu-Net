import torchvision.utils as vutils
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from PIL import Image
import colorsys
import cv2
def write_2images(image_outputs, display_image_num, image_directory, postfix):
    __write_images(image_outputs[:], display_image_num, '%s/gen_%s.png' % (image_directory, postfix))

def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def collate_func(batch_dic):
    output={}
    #for k in ['labels','ins_masks']:
    for k in ['cate_labels', 'ins_labels','ins_ind_labels']:
        output[k]=[dic[k] if dic[k] is not None else None for dic in batch_dic]
    output['image']=torch.stack([dic['image'] for dic in batch_dic])
    return output

def convert_labels(arr):
    w,h=arr.shape
    output=np.zeros([w,h,3],dtype=np.uint8)
    for i in np.unique(arr):
        if i ==0:continue
        output[arr==i]=[random.randint(64,255),random.randint(64,255),random.randint(64,255)]
    return output

def _imageshow(image,pred,gt,unpaired_pred_ind,unpaired_gt_ind,title=None,cmap='tab20b'):
    plt.rcParams['figure.figsize'] = (8.0, 6.0)  # 设置figure_size尺寸
    plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300

    plt.subplot(231)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(convert_labels(pred))
    plt.title(f'pred_num {len(np.unique(pred))}')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(convert_labels(gt))
    plt.title(f'gt_num {len(np.unique(gt))}')
    plt.axis('off')
    plt.subplot(224)
    plt.axis('off')
    unmatched_pred=np.zeros_like(pred)
    for i in np.unique(unpaired_pred_ind):
        if i ==0:
            continue
        unmatched_pred[pred==i]=i
    plt.subplot(235)
    plt.imshow(convert_labels(unmatched_pred))
    plt.title(f'unmatched_pred {len(np.unique(unpaired_pred_ind))}')
    plt.axis('off')
    unmatched_gt=np.zeros_like(gt)
    for i in np.unique(unpaired_gt_ind):
        if i ==0:
            continue
        unmatched_gt[gt==i]=i
    plt.subplot(236)
    plt.imshow(convert_labels(unmatched_gt))
    plt.title(f'unmatched_gt {len(np.unique(unpaired_gt_ind))}')
    plt.axis('off')
    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.99, left=0.01, hspace=0.1, wspace=0.1)
    plt.tight_layout()
    #plt.savefig('D:/{}.png'.format(time.time()))
    plt.show()


def random_colors(N, bright=True):
    """Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def _imagesave(image, mask, label,save_path):
    colors={
        1: [255, 0, 0],
        2:[0, 255, 0],
        3: [0, 0, 255],
        4:  [255, 255, 0],
        5 : [255, 165, 0]
    }

    inst_rng_colors = random_colors(len(np.unique(mask)))
    inst_rng_colors = np.clip(inst_rng_colors,0,255) * 255
    inst_rng_colors = inst_rng_colors.astype(np.uint8).tolist()

    image=image.copy()
    if image.shape[2]==4:
        image=image[:,:,0:3].copy()


    for ui in np.unique(mask):
        if ui == 0: continue
        binary = ((mask == ui) * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if label is not None:
            image = cv2.drawContours(image, contours, -1, tuple(colors[label[ui-1]]), 2)
        else:
            image = cv2.drawContours(image, contours, -1, tuple(inst_rng_colors[ui-1]), 2)

    Image.fromarray(image).save(save_path)
