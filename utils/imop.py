import numpy as np
from scipy.ndimage.interpolation import shift
import cv2
from collections import Counter
import math
from scipy import ndimage
def get_ins_info(seg_mask,method='bbox'):
    methods=['bbox','circle','area']
    assert method in methods, f'instance segmentation information should in {methods}'
    if method=='circle':
        contours, hierachy = cv2.findContours((seg_mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        (center_w, center_h), EC_radius = cv2.minEnclosingCircle(contours[0])
        return center_w,center_h,EC_radius*2,EC_radius*2
    elif method=='bbox':
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(np.array(seg_mask).astype(np.uint8))
        center_w = bbox_x + bbox_w / 2
        center_h = bbox_y + bbox_h / 2
        return center_w, center_h, bbox_w, bbox_h
    elif method=='area':
        center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
        equal_diameter=(np.sum(seg_mask)/3.1415)**0.5*2
        return center_w,center_h,equal_diameter,equal_diameter
    else:
        raise NotImplementedError

def gaussian_radius(det_size, min_overlap=0.7):

    #https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)

def gaussian_radius_new(det_size, min_overlap=0.7):
    #https://github.com/princeton-vl/CornerNet/blob/master/sample/utils.py
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)

    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1.):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius):
    diameter = float(2 * (int(radius)+1) + 1)
    gaussian = gaussian2D((diameter, diameter), sigma = radius/3)#gaussian_2d_kernel(int(diameter),radius/3)#
    coord_w, coord_h = center
    height, width = heatmap.shape
    temp=np.zeros((height,width), dtype=np.float)
    temp = insert_image(temp, gaussian, coord_h, coord_w)
    np.maximum(heatmap, temp, out=heatmap)
    return temp

def gaussian_2d_kernel(kernel_size=3, sigma=0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2
    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)

    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
    return kernel

def insert_image(img, kernel,h,w):
    ks=kernel.shape[0]
    if ks !=0:
        half_ks=ks//2
        img=np.pad(img,((half_ks,half_ks),(half_ks,half_ks)))
        img[h:h+ks,w:w+ks]=kernel
        return img[half_ks:-half_ks,half_ks:-half_ks]
    else:
        img[h:h+1,w:w+1]=kernel
        return img

def label_relaxation(msk, border_window=3):
    msk=msk.astype(np.float)
    border=border_window//2
    output=np.zeros_like(msk)
    for i in range(-border,border+1,1):
        for j in range(-border, border + 1, 1):
            output+=shift(msk,shift=[i,j],cval=0)
    output/=border_window**2
    return output

def ensemble_img(ins_list,cate_list,score_list):
    N,w,h=ins_list.shape
    output_img=np.zeros((w,h),dtype=np.int16)
    #print(ins_list.shape,cate_list.shape,score_list.shape)
    for i in range(N):
        ins_num = i+1
        ins_=ins_list[i]
        score_=score_list[i]
        if np.sum(np.logical_and(output_img>0,ins_))==0:
            output_img=np.where(output_img>0,output_img,ins_*ins_num)
        else:
            compared_num,_ = Counter((output_img*ins_).flatten()).most_common(2)[1]
            assert compared_num>0
            #print( Counter((output_img*ins_).flatten()).most_common(2))
            compared_num=int(compared_num)
            compared_score=score_list[compared_num-1]
            if np.sum(np.logical_and(output_img==compared_num,ins_))/np.sum(np.logical_or(output_img==compared_num,ins_))>0.5:
                if compared_score>score_:
                    pass
                else:
                    output_img[output_img==compared_num]=0
                    output_img=np.where(output_img>0,output_img,ins_*ins_num)
            else:
                output_img = np.where(output_img > 0, output_img, ins_ * ins_num)
    return output_img

def gaussian_radius1(det_size,min_overlap=0.7):
    height, width = det_size
    ra=(1-min_overlap**0.5)/2**0.5 *height
    rb=(1-min_overlap**0.5)/2**0.5 *width
    return ra,rb

def gaussian2D1(shape, sigmah=1.,sigmaw=1.):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x / (2 * sigmaw * sigmaw)+y * y / (2 * sigmah * sigmah)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian1(heatmap, center,rw, rh):
    diameterw = float(2 * (int(rw)+1) + 1)
    diameterh = float(2 * (int(rh)+1) + 1)
    gaussian = gaussian2D1((diameterh, diameterw), sigmah= rh / 3, sigmaw= rw / 3)#gaussian_2d_kernel(int(diameter),radius/3)#
    coord_w, coord_h = center
    height, width = heatmap.shape
    temp=np.zeros((height,width), dtype=np.float)
    temp = insert_image(temp, gaussian, coord_h, coord_w)
    np.maximum(heatmap, temp, out=heatmap)
    return temp