import torch.utils.data as data
import os
from torchvision import transforms
from tifffile import TiffFile
from PIL import Image
import albumentations as A
import torch
import numpy as np

from skimage.util.shape import view_as_windows
import scipy.io as scio
from .io import make_dataset
from ._aug import get_augmentation
from .imop import get_ins_info,gaussian_radius,draw_gaussian

import matplotlib.pyplot as plt

class NucleiDataset(data.Dataset):
    def __init__(self,config,seed, is_train,output_stride=4):
        stain_norm=config['stainnorm']
        data_root=config['dataroot']
        img_size=256
        self.grid_size=img_size//output_stride

        if not config['image_norm_mean']:
            _mean=(0.5,0.5,0.5)
            _std=(0.5,0.5,0.5)
        else:
            _mean = tuple(config['image_norm_mean'])
            _std = tuple(config['image_norm_std'])
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(_mean, _std)])
        print(f'Stain Norm Type: {stain_norm}, Trans mean std: {_mean}, {_std}.')

        self.phase='train' if is_train else 'test'

        self.img_dir_path = os.path.join(data_root,self.phase, 'Images') if stain_norm is None else os.path.join(data_root, self.phase, f'Images_{stain_norm}')
        self.img_paths = sorted(make_dataset(self.img_dir_path))
        self.gt_dir_path = os.path.join(data_root,self.phase, 'Labels')
        self.gt_paths = sorted(make_dataset(self.gt_dir_path))
        self.num_class=config['model']['num_classes']
        self.use_class= config['model']['num_classes']>2 and 'CoNSeP' in config['dataroot']
        print(f'use classes {self.use_class}')
        self.images,self.masks,self.labels=self.load_crop_data(self.img_paths,self.gt_paths)

        self._size =self.images.shape[0]
        self.setup_augmentor(seed)

    def load_crop_data(self, img_paths, gt_paths, patch_size=384, stride=128):
        out_imgs = []
        out_masks = []
        out_labels = []
        resize = A.Resize(p=1, height=1024, width=1024)
        for ip, mp in zip(img_paths, gt_paths):
            assert os.path.basename(ip)[:-4] == os.path.basename(mp)[:-4]
            if '.tif' in ip:
                with TiffFile(ip) as t:
                    im=t.asarray()
            else:
                im = np.array(Image.open(ip).convert('RGB'))
            matfile=scio.loadmat(mp)
            mk=matfile['inst_map'].astype(np.int16)
            if self.use_class:
                lbl=matfile['inst_type'][:,0].astype(np.uint8)
                lbl[lbl == 4] = 3
                lbl[lbl == 5] = 4
                lbl[lbl == 6] = 4
                lbl[lbl == 7] = 4
            else:
                lbl=[1]*(np.max(mk))
            augmented = resize(image=im, mask=mk)
            im = augmented['image']
            mk = augmented['mask']
            ims = view_as_windows(im, (patch_size, patch_size, 3), (stride, stride, 3)).reshape((-1, patch_size, patch_size, 3))
            mks = view_as_windows(mk, (patch_size, patch_size), (stride, stride)).reshape((-1, patch_size, patch_size))
            out_imgs.append(ims)
            out_masks.append(mks)
            for idx in range(mks.shape[0]):
                tmk=mks[idx]
                olabel = {}
                for ui in np.unique(tmk):
                    if ui==0:
                        continue
                    olabel[ui]=lbl[ui-1]
                out_labels.append(olabel)

        out_imgs = np.concatenate(out_imgs)
        out_masks = np.concatenate(out_masks)
        assert len(out_imgs.shape) == 4 and out_imgs.dtype == np.uint8

        print(f'processed data with size {len(out_imgs)} & {len(out_labels)}')
        return out_imgs, out_masks, out_labels

    def setup_augmentor(self, seed):
        self.shape_augs, self.input_augs = get_augmentation(self.phase, seed)

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        label_dic = self.labels[index]

        shape_augs = self.shape_augs.to_deterministic()
        img = shape_augs.augment_image(img)
        masks = shape_augs.augment_image(mask)

        input_augs = self.input_augs.to_deterministic()
        img = input_augs.augment_image(img)

        cate_labels=[]
        ins_labels=[]

        for i, ui in enumerate(np.unique(masks)):
            if ui ==0:
                assert i==ui
                continue
            tmp_mask=masks==ui
            label=label_dic[ui]
            ins_labels.append(((tmp_mask)*1).astype(np.int32))
            cate_labels.append(label)

        if len(cate_labels)>0:
            cate_labels, ins_labels, ins_ind_labels= self.process_label(np.array(cate_labels), np.array(ins_labels))
            cate_labels=torch.from_numpy(np.array(cate_labels)).float()
            ins_labels=torch.from_numpy(ins_labels)
            ins_ind_labels=torch.from_numpy(ins_ind_labels).bool()
        else:
            cate_labels=torch.from_numpy(np.zeros([self.num_class - 1,64,64])).float()
            ins_labels=None
            ins_ind_labels=None
        image=self.transform(img)
        output={'image': image, 'cate_labels':cate_labels, 'ins_labels':ins_labels,'ins_ind_labels':ins_ind_labels}
        return output

    def process_label(self, gt_labels_raw, gt_masks_raw, iou_threshold=0.3, tau=0.5):
        w,h=256,256

        cate_label = np.zeros([self.num_class - 1, self.grid_size, self.grid_size], dtype=np.float)
        ins_label = np.zeros([self.grid_size ** 2, w, h], dtype=np.int16)
        ins_ind_label = np.zeros([self.grid_size ** 2], dtype=np.bool)
        if gt_masks_raw is not None:
            gt_labels = gt_labels_raw
            gt_masks = gt_masks_raw
            for seg_mask, gt_label in zip(gt_masks, gt_labels):
                center_w, center_h, width, height = get_ins_info(seg_mask, method='bbox')
                radius = max(gaussian_radius((width, height), iou_threshold), 0)
                coord_h = int((center_h / h) / (1. / self.grid_size))
                coord_w = int((center_w / w) / (1. / self.grid_size))
                temp = draw_gaussian(cate_label[gt_label - 1], (coord_w, coord_h), (radius / 4))
                non_zeros = (temp > tau).nonzero()
                label = non_zeros[0] * self.grid_size + non_zeros[1]  # label = int(coord_h * grid_size + coord_w)#
                ins_label[label, :, :] = seg_mask
                ins_ind_label[label] = True
        ins_label=np.stack(ins_label[ins_ind_label],0)
        return cate_label,ins_label,ins_ind_label

    def __len__(self):
        return self._size

class PannukeDataset(data.Dataset):
    """
    img_path: original image
    masks: one-hot masks
    GT: tiff mask, one channel denote one instance
    """
    def __init__(self, data_root, is_train,seed=888,fold=1,output_stride=4):
        self.grid_size=256//output_stride

        self.images,self.labels,self.masks= self.load_pannuke(data_root,fold)
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        self.num_class=6
        self.A_size =self.images.shape[0]

        self.mode='train' if is_train else "test"
        self.setup_augmentor(seed)

    def setup_augmentor(self, seed):
        self.shape_augs, self.input_augs = get_augmentation(self.mode, seed)

    def load_pannuke(self, data_root,fold=1):
        out_labels = []
        out_masks = []
        out_imgs=np.load(os.path.join(data_root,f'images/fold{fold}/images.npy')).astype(np.uint8)#(2523, 256, 256, 3)
        masks=np.load(os.path.join(data_root,f'masks/fold{fold}/masks.npy')).astype(np.int16)#(2523, 256, 256, 6)
        for i in range(masks.shape[0]):
            tmask=masks[i]
            olabel={}
            omask=np.zeros((256,256),dtype=np.int16)
            for j in range(5):
                ids=np.unique(tmask[:,:,j])
                if len(ids) ==1:
                    continue
                else:
                    for id in ids:
                        if id==0:continue
                        omask[tmask[:,:,j]==id]=id
                        olabel[id]=j+1
            out_masks.append(omask)
            out_labels.append(olabel)
        out_masks = np.stack(out_masks,0)
        assert len(out_imgs.shape) == 4 and out_imgs.dtype == np.uint8
        assert len(out_masks.shape) == 3 and out_masks.dtype == np.int16, f'{out_masks.shape}, {out_masks.dtype}'
        assert out_masks.shape[0]==out_imgs.shape[0] and out_imgs.shape[0]==len(out_labels)
        print(f'processed data with size {len(out_imgs)}')
        return out_imgs, out_labels, out_masks

    def __getitem__(self, index):
        img = self.images[index]
        mask = self.masks[index]
        label_dic = self.labels[index]

        shape_augs = self.shape_augs.to_deterministic()
        img = shape_augs.augment_image(img)
        masks = shape_augs.augment_image(mask)

        input_augs = self.input_augs.to_deterministic()
        img = input_augs.augment_image(img)

        cate_labels=[]
        ins_labels=[]

        for i, ui in enumerate(np.unique(masks)):
            if ui ==0:
                assert i==ui
                continue
            tmp_mask=masks==ui
            label=label_dic[ui]
            ins_labels.append(((tmp_mask)*1).astype(np.int32))
            cate_labels.append(label)

        if len(cate_labels)>0:
            cate_labels, ins_labels, ins_ind_labels= self.process_label(np.array(cate_labels), np.array(ins_labels))
            cate_labels=torch.from_numpy(np.array(cate_labels)).float()
            ins_labels=torch.from_numpy(ins_labels)
            ins_ind_labels=torch.from_numpy(ins_ind_labels).bool()
        else:
            cate_labels=torch.from_numpy(np.zeros([self.num_class - 1,self.grid_size,self.grid_size])).float()
            ins_labels=None
            ins_ind_labels=None
        image=self.transform(img)
        output={'image': image, 'cate_labels':cate_labels, 'ins_labels':ins_labels,'ins_ind_labels':ins_ind_labels}
        return output

    def process_label(self, gt_labels_raw, gt_masks_raw, iou_threshold=0.3, tau=0.5):
        w,h=256,256
        cate_label = np.zeros([self.num_class - 1, self.grid_size, self.grid_size], dtype=np.float)
        ins_label = np.zeros([self.grid_size ** 2, w, h], dtype=np.int16)
        ins_ind_label = np.zeros([self.grid_size ** 2], dtype=np.bool)
        if gt_masks_raw is not None:
            gt_labels = gt_labels_raw
            gt_masks = gt_masks_raw
            for seg_mask, gt_label in zip(gt_masks, gt_labels):
                center_w, center_h, width, height = get_ins_info(seg_mask, method='bbox')
                radius = max(gaussian_radius((width, height), iou_threshold), 0)
                coord_h = int((center_h / h) / (1. / self.grid_size))
                coord_w = int((center_w / w) / (1. / self.grid_size))
                temp = draw_gaussian(cate_label[gt_label - 1], (coord_w, coord_h), (radius / 4))
                non_zeros = (temp > tau).nonzero()
                label = non_zeros[0] * self.grid_size + non_zeros[1]  # label = int(coord_h * grid_size + coord_w)#
                cate_label[gt_label - 1, coord_h, coord_w] = 1
                label = int(coord_h * self.grid_size + coord_w)  #
                ins_label[label, :, :] = seg_mask
                ins_ind_label[label] = True
        ins_label=np.stack(ins_label[ins_ind_label],0)
        return cate_label, ins_label, ins_ind_label

    def __len__(self):
        return self.A_size