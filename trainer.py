from models.PointNuNet import PointNuNet
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
import os
import torch.nn.functional as F
from utils.fmix import sample_mask
from utils.matrix_nms import matrix_nms
from torch.cuda.amp import autocast as autocast,GradScaler
from losses import BinaryDiceLoss


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.num_class = config['model']['num_classes']
        frozen_stages = config['model']['frozen_stages']
        norm_eval = config['model']['norm_eval']
        backbone = config['model']['backbone']
        pretrained_base = config['model']['pretrain']
        seg_feat_channels = config['model']['seg_feat_channels']
        stacked_convs = config['model']['stacked_convs']
        output_stride=config['model']['output_stride']

        self.ins_out_channels = config['model']['ins_out_channels']
        self.kernel_size= config['model']['kernel_size']

        if torch.cuda.device_count() == 1:
            self.model = PointNuNet(nclass=self.num_class, backbone=backbone, pretrained_base=pretrained_base,
                              frozen_stages=frozen_stages, norm_eval=norm_eval,
                              seg_feat_channels=seg_feat_channels, stacked_convs=stacked_convs,
                              ins_out_channels=self.ins_out_channels,kernel_size=self.kernel_size,output_stride=output_stride)
        else:
            print('multi GPU training detection')
            self.model=nn.DataParallel(PointNuNet(nclass=self.num_class, backbone=backbone, pretrained_base=pretrained_base,
                              frozen_stages=frozen_stages, norm_eval=norm_eval,
                              seg_feat_channels=seg_feat_channels, stacked_convs=stacked_convs,
                              ins_out_channels=self.ins_out_channels,kernel_size=self.kernel_size,output_stride=output_stride))

        self.models_params = list(self.model.parameters())

        if config['train']['optim'] == 'adamw':
            self.opt = torch.optim.AdamW([p for p in self.models_params if p.requires_grad], lr=config['train']['lr'],
                                         betas=(config['train']['beta1'], config['train']['beta2']),
                                         weight_decay=config['train']['weight_decay'])
        elif config['train']['optim'] == 'adam':
            self.opt = torch.optim.Adam([p for p in self.models_params if p.requires_grad], lr=config['train']['lr'],
                                         betas=(config['train']['beta1'], config['train']['beta2']),
                                         weight_decay=config['train']['weight_decay'])
        else:
            raise NotImplementedError


        if config['train']['lr_policy'] == 'multistep':
            max_epoch=config['train']['max_epoch']
            print(f'Multi step scheduler decay at {int(0.8*max_epoch)} and {int(0.9*max_epoch)} with gamma 0.1')
            self.scheduler = lr_scheduler.MultiStepLR(self.opt,milestones=[int(0.8*max_epoch),int(0.9*max_epoch)], gamma=0.1)
        elif config['train']['lr_policy'] == 'step':
            self.scheduler = lr_scheduler.StepLR(self.opt,step_size = 1, gamma = 0.96)
        elif config['train']['lr_policy']=='cosineannwarm':
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.opt,T_0=5,T_mult=2)
        else:
            raise NotImplementedError
        self.use_mixed=config['train']['use_mixed']
        if self.use_mixed:
            self.scaler = GradScaler()
        self.lambda_ins=config['train']['lambda_ins']
        self.lambda_cate=config['train']['lambda_cate']
        self.mask_rescoring=config['mask_rescoring']
        self.ins_loss=BinaryDiceLoss()
        self.ce_loss=nn.BCEWithLogitsLoss(reduction='sum')

    def points_nms(self, heat, kernel=3):
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel)//2)
        keep = (hmax == heat).float()
        return heat * keep

    def prediction_fast(self, image):

        feature_preds, kernel_preds, cate_preds = self.model(image)
        N, E, h, w = feature_preds.shape
        assert N==1

        kernel_preds = kernel_preds[0].permute(1, 2, 0).view(-1, E)
        cate_preds = self.points_nms(cate_preds.sigmoid(), kernel=3)
        cate_preds = cate_preds.permute(0, 2, 3, 1).view(-1, self.num_class - 1)
        inds = (cate_preds > 0.4)

        cate_scores = cate_preds[inds]

        inds = inds.nonzero(as_tuple=False)

        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        I, N = kernel_preds.shape
        if I==0:
            return None
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(feature_preds, kernel_preds,stride=1).squeeze(0)

        seg_masks= seg_preds.sigmoid() > 0.5# (N, 64 ** 2, h, w)
        sum_masks = seg_masks.sum((1, 2)).float()

        # min area filter.
        keep = sum_masks > 4
        if keep.sum() == 0:
            return None
        seg_masks = seg_masks[keep, ...]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        return seg_masks, cate_labels, cate_scores

    def prediction_single(self, image, use_nms=True, nms_pre=500, max_per_img=200,  score_thr=0.4, update_thr=0.2):
        self.eval()
        feature_preds, kernel_preds, cate_preds = self.model(image)
        self.train()

        N, E, h, w = feature_preds.shape
        assert N==1

        kernel_preds = kernel_preds[0].permute(1, 2, 0).view(-1, E)
        cate_preds = self.points_nms(cate_preds.sigmoid(), kernel=3) if not use_nms else cate_preds.sigmoid()
        cate_preds = cate_preds.permute(0, 2, 3, 1).view(-1, self.num_class - 1)
        inds = (cate_preds > score_thr)
        cate_scores = cate_preds[inds]

        inds = inds.nonzero(as_tuple=False)
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        I, N = kernel_preds.shape
        if I==0:
            return None
        kernel_preds = kernel_preds.view(I, N, 1, 1)

        seg_preds = F.conv2d(feature_preds, kernel_preds,stride=1).squeeze(0)
        seg_masks= seg_preds.sigmoid() > 0.5# (N, 64 ** 2, h, w)
        sum_masks = seg_masks.sum((1, 2)).float()

        seg_scores = (seg_preds.sigmoid() * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # min area filter.

        keep = sum_masks > 4

        if keep.sum() == 0:
            return None
        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]
        if use_nms:
            # sort and keep top nms_pre
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > nms_pre:
                sort_inds = sort_inds[:nms_pre]
            seg_masks = seg_masks[sort_inds, :, :]
            seg_preds = seg_preds[sort_inds, :, :]
            sum_masks = sum_masks[sort_inds]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

            cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=10, sum_masks=sum_masks)
            keep = cate_scores >= update_thr
            if keep.sum() == 0:
                return None
            seg_preds = seg_preds[keep, :, :]
            cate_scores = cate_scores[keep]
            cate_labels = cate_labels[keep]

            # sort and keep top_k
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > max_per_img:
                sort_inds = sort_inds[:max_per_img]
            seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

        seg_masks = seg_preds.sigmoid() > 0.5
        return seg_masks, cate_labels, cate_scores

    def prediction(self, image, score_thr=0.4 ,use_nms=True, max_per_img=200, nms_pre=500, update_thr=0.2):
        self.eval()
        feature_preds, kernel_preds, cate_preds = self.model(image)
        self.train()

        N, E, h, w = feature_preds.shape
        feature_preds = feature_preds[0].view(-1, h, w).unsqueeze(0)
        kernel_preds = kernel_preds[0].permute(1, 2, 0).view(-1, E)

        cate_preds = self.points_nms(cate_preds.sigmoid(), kernel=3) if not use_nms else cate_preds.sigmoid()

        cate_preds = cate_preds.permute(0, 2, 3, 1).view(-1, self.num_class - 1)

        inds = (cate_preds > score_thr)
        cate_scores = cate_preds[inds]

        inds = inds.nonzero(as_tuple=False)
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        I, N = kernel_preds.shape
        if I==0:
            return None
        kernel_preds = kernel_preds.view(I, N, 1, 1)

        seg_preds = F.conv2d(feature_preds, kernel_preds,stride=1).squeeze(0)
        seg_masks= seg_preds.sigmoid() > 0.5# (N, 64 ** 2, h, w)
        sum_masks = seg_masks.sum((1, 2)).float()

        seg_scores = (seg_preds.sigmoid() * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # min area filter.
        keep = sum_masks > 4
        if keep.sum() == 0:
            return None
        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        if use_nms:
            # sort and keep top nms_pre
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > nms_pre:
                sort_inds = sort_inds[:nms_pre]
            seg_masks = seg_masks[sort_inds, :, :]
            seg_preds = seg_preds[sort_inds, :, :]
            sum_masks = sum_masks[sort_inds]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

            cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=10, sum_masks=sum_masks)
            keep = cate_scores >= update_thr
            if keep.sum() == 0:
                return None
            seg_preds = seg_preds[keep, :, :]
            cate_scores = cate_scores[keep]
            cate_labels = cate_labels[keep]

            # sort and keep top_k
            sort_inds = torch.argsort(cate_scores, descending=True)
            if len(sort_inds) > max_per_img:
                sort_inds = sort_inds[:max_per_img]
            seg_preds = seg_preds[sort_inds, :, :]
            cate_scores = cate_scores[sort_inds]
            cate_labels = cate_labels[sort_inds]

        seg_masks = seg_preds.sigmoid() > 0.5
        return seg_masks, cate_labels, cate_scores

    def re_index(self,lists,index):
        output=[]
        for i in index:
            output.append(lists[i])
        return output

    def seg_updata_FMIX(self, train_data,alpha=1,decay_power=3,size=(256, 256), max_soft=0.0, reformulate=False):

        image,target_ins_a, target_ins_ind_a, target_cate_a= train_data['image'],train_data['ins_labels'],train_data['ins_ind_labels'],train_data['cate_labels']

        lam, mask = sample_mask(alpha, decay_power, size, max_soft, reformulate)

        index = torch.randperm(image.size(0)).to(image.device)

        mask = torch.from_numpy(mask).float().to(image.device)

        target_cate_b=self.re_index(target_cate_a,index.cpu().numpy().tolist())

        target_ins_b=self.re_index(target_ins_a,index.cpu().numpy().tolist())

        target_ins_ind_b=self.re_index(target_ins_ind_a,index.cpu().numpy().tolist())

        image=mask * image+(1 - mask) * image[index]

        if self.use_mixed:
            self.opt.zero_grad()
            with autocast():
                feature_preds, kernel_preds, cate_preds = self.model(image)
                losses_a = self.losses(feature_preds, kernel_preds, cate_preds, target_ins_a,target_ins_ind_a, target_cate_a)
                losses_b = self.losses(feature_preds, kernel_preds, cate_preds, target_ins_b,target_ins_ind_b, target_cate_b)
            losses_ins = (losses_a['loss_ins'] * lam + losses_b['loss_ins'] * (1. - lam)) * self.lambda_ins
            losses_cate = (losses_a['loss_cate'] * lam + losses_b['loss_cate'] * (1. - lam)) * self.lambda_cate
            self.loss_seg_total = losses_ins + losses_cate
            self.scaler.scale(self.loss_seg_total).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.zero_grad()
            feature_preds, kernel_preds, cate_preds = self.model(image)
            losses_a = self.losses(feature_preds, kernel_preds, cate_preds, target_ins_a, target_ins_ind_a, target_cate_a)
            losses_b = self.losses(feature_preds, kernel_preds, cate_preds, target_ins_b,target_ins_ind_b,  target_cate_b)
            losses_ins = (losses_a['loss_ins'] * lam + losses_b['loss_ins'] * (1. - lam)) * self.lambda_ins
            losses_cate = (losses_a['loss_cate'] * lam + losses_b['loss_cate'] * (1. - lam)) * self.lambda_cate
            self.loss_seg_total = losses_ins + losses_cate
            self.loss_seg_total.backward()
            self.opt.step()
        return losses_ins.item(),losses_cate.item() ,0

    def seg_updata(self, train_data):
        image,target_ins, target_ins_ind, target_cate= train_data['image'],train_data['ins_labels'],train_data['ins_ind_labels'],train_data['cate_labels']
        if self.use_mixed:
            self.opt.zero_grad()
            with autocast():
                feature_preds, kernel_preds, cate_preds = self.model(image)
                losses = self.losses(feature_preds, kernel_preds, cate_preds, target_ins, target_ins_ind, target_cate)
            self.loss_seg_total = losses['loss_ins'] * self.lambda_ins + losses['loss_cate'] * self.lambda_cate
            self.scaler.scale(self.loss_seg_total).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.zero_grad()
            feature_preds, kernel_preds, cate_preds = self.model(image)
            losses = self.losses(feature_preds, kernel_preds, cate_preds, target_ins,target_ins_ind, target_cate)
            self.loss_seg_total = losses['loss_ins'] * self.lambda_ins + losses['loss_cate'] * self.lambda_cate
            self.loss_seg_total.backward()
            self.opt.step()
        return losses['loss_ins'].item() * self.lambda_ins, losses['loss_cate'].item() * self.lambda_cate, 0

    def losses(self, feature_preds, kernel_preds, cate_preds, gt_ins,gt_ins_ind,gt_cates):
        loss_ins=[]
        loss_cate=[]
        N, _, h, w = feature_preds.shape
        for batch_idx in range(N):
            feature_pred=feature_preds[batch_idx]
            kernel_pred=kernel_preds[batch_idx]
            cate_pred=cate_preds[batch_idx]
            gt_cate = gt_cates[batch_idx].float()
            loss_cate.append(self.local_focal(cate_pred.sigmoid(), gt_cate))
            if gt_ins[batch_idx] is not None:
                gt_mask = gt_ins[batch_idx].float()
                gt_mask_ind=gt_ins_ind[batch_idx]
                kernel_pred = kernel_pred.permute(1, 2, 0).contiguous().view(-1, self.ins_out_channels * self.kernel_size * self.kernel_size)
                kernel_pred = torch.cat([kernel_pred[gt_mask_ind]], 0).view(-1, self.ins_out_channels, self.kernel_size, self.kernel_size)
                ins_pred = F.conv2d(feature_pred.unsqueeze(0), kernel_pred, stride=1).view(-1, h, w)
                loss_ins.append(self.ins_loss(ins_pred, gt_mask))

            else:
                continue
        return {
            'loss_ins': torch.stack(loss_ins).mean(),
            'loss_cate':  torch.stack(loss_cate).mean(),
        }


    def save(self, snapshot_dir, epoch):
        if isinstance(epoch,int):
            model_name = os.path.join(snapshot_dir, 'model_%04d.pt' % (epoch + 1))
            opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
            torch.save({'seg': self.model.state_dict()}, model_name)
            torch.save({'seg': self.opt.state_dict()}, opt_name)
        elif isinstance(epoch,str):
            model_name = os.path.join(snapshot_dir, 'model_%s.pt' % epoch)
            opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
            torch.save({'seg': self.model.state_dict()}, model_name)
            torch.save({'seg': self.opt.state_dict()}, opt_name)


    def local_focal(self, pred, gt):
        """
        focal loss copied from CenterNet, modified version focal loss
        change log: numeric stable version implementation
        """
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)
        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        #print(num_pos)

        if num_pos == 0:
            loss =  - neg_loss
        else:
            loss = - (pos_loss + neg_loss) / num_pos
        return loss


    def find_local_peak(self, heat, kernel=3):
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel)//2)
        keep = (hmax == heat).float()
        return heat * keep

