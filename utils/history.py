def get_gt_single(self, im, gt_instances):
    gt_labels_raw = gt_instances['labels'][im]
    gt_masks_raw = gt_instances['ins_masks'][im]
    device = gt_labels_raw[0].device
    ins_label = torch.zeros([64 ** 2, 256, 256], dtype=torch.uint8, device=device)
    # NOTE: gt_labels_raw between 0~79.
    cate_label = torch.zeros([64, 64], dtype=torch.int64, device=device).fill_(0)
    ins_ind_label = torch.zeros([64 ** 2], dtype=torch.bool, device=device)
    gt_labels = gt_labels_raw
    gt_masks = gt_masks_raw.cpu().numpy().astype(np.uint8)

    for seg_mask, gt_label in zip(gt_masks, gt_labels):
        center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
        coord_w = int((center_w / 256) // (1. / 64))
        coord_h = int((center_h / 256) // (1. / 64))
        cate_label[coord_h, coord_w] = gt_label
        seg_mask = torch.Tensor(seg_mask)
        label = int(coord_h * 64 + coord_w)
        ins_label[label, :, :] = seg_mask
        ins_ind_label[label] = True

    return ins_label, cate_label, ins_ind_label


def _get_gt(self, gt_instances):
    ins_label_list, cate_label_list, ins_ind_label_list = [], [], []
    for im in range(gt_instances['image'].size(0)):
        _ins_label, _cate_label, _ins_ind_label = self.get_gt_single(im, gt_instances)
        ins_label_list.append(_ins_label.float())
        cate_label_list.append(_cate_label.long())
        ins_ind_label_list.append(_ins_ind_label)
    return ins_label_list, cate_label_list, ins_ind_label_list


def losses(self, ins_preds, cate_preds, gt_instances):
    ins_label_list, cate_label_list, ins_ind_label_list = self._get_gt(gt_instances)
    # ins
    ins_labels = [torch.cat([ins_labels_img[ins_ind_labels_img]], 0) for ins_labels_img, ins_ind_labels_img in
                  zip(ins_label_list, ins_ind_label_list)]

    ins_preds = [torch.cat([ins_preds_img[ins_ind_labels_img]], 0) for ins_preds_img, ins_ind_labels_img in
                 zip(ins_preds, ins_ind_label_list)]
    # dice loss
    loss_ins = []
    for input, target in zip(ins_preds, ins_labels):
        if input.size()[0] == 0:
            continue
        input = torch.sigmoid(input)
        loss_ins.append(dice_loss(input, target))
    loss_ins = torch.cat(loss_ins).mean()
    loss_ins = loss_ins * self.ins_loss_weight

    # cate
    flatten_cate_labels = torch.cat([cate_labels.flatten() for cate_labels in cate_label_list], 0).reshape(-1, 1)

    flatten_cate_preds = cate_preds.permute(0, 2, 3, 1).reshape(-1, 1)

    loss_cate = self.ce_loss(flatten_cate_preds, flatten_cate_labels)

    return {
        'loss_ins': loss_ins,
        'loss_cate': loss_cate
    }


def make_rotate_affine_matrix(angle):
    assert len(angle.size()) == 2 and angle.size(1) == 3, f'{len(angle.size())}, {angle.size(1)}'
    B, _ = angle.size()
    angle = angle.sigmoid()
    angle = angle * math.pi
    rX, rY, rZ = torch.split(angle, [1, 1, 1], dim=-1)  # N*1
    rX = rX.squeeze()
    rY = rY.squeeze()
    rZ = rZ.squeeze()
    matrix = torch.zeros((B, 3, 4))
    matrix[:, 0, 0] = torch.cos(rY) * torch.cos(rZ)
    matrix[:, 0, 1] = - torch.cos(rY) * torch.sin(rZ)
    matrix[:, 0, 2] = torch.sin(rY)
    matrix[:, 1, 0] = torch.sin(rX) * torch.sin(rY) * torch.cos(rZ) + torch.cos(rX) * torch.sin(rZ)
    matrix[:, 1, 1] = -torch.sin(rX) * torch.sin(rY) * torch.sin(rZ) + torch.cos(rX) * torch.cos(rZ)
    matrix[:, 1, 2] = -torch.sin(rX) * torch.cos(rY)
    matrix[:, 2, 0] = -torch.cos(rX) * torch.sin(rY) * torch.cos(rZ) + torch.sin(rX) * torch.sin(rZ)
    matrix[:, 2, 1] = torch.cos(rX) * torch.sin(rY) * torch.sin(rZ) + torch.sin(rX) * torch.cos(rZ)
    matrix[:, 2, 2] = torch.cos(rX) * torch.cos(rY)
    return matrix.cuda()

# project_pred,angle_pred=torch.split(kernel_pred, [c,3],dim=1)
# theta = make_rotate_affine_matrix(angle_pred)

# grid = F.affine_grid(theta, torch.Size((project_pred.size(0), 1, 256, 256, 256)))

# feature_pred=feature_pred.unsqueeze(0).unsqueeze(0)# 1*1*D*H*W
# rotated_feature=F.grid_sample(feature_pred,grid,align_corners=True)# N*1*D*H*W
# copy from: https://github.com/wuhuikai/FastFCN/blob/master/encoding/nn/customize.py
class JPU123(nn.Module):
    def __init__(self, in_channels, width=256, norm_layer=nn.BatchNorm2d, **kwargs):
        super(JPU123, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))

        self.dilation1 = nn.Sequential(
            SeparableConv2d(4 * width, width, 3, padding=1, dilation=1, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.dilation2 = nn.Sequential(
            SeparableConv2d(4 * width, width, 3, padding=2, dilation=2, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.dilation3 = nn.Sequential(
            SeparableConv2d(4 * width, width, 3, padding=4, dilation=4, bias=False),
            norm_layer(width),
            nn.ReLU(True))
        self.dilation4 = nn.Sequential(
            SeparableConv2d(4 * width, width, 3, padding=8, dilation=8, bias=False),
            norm_layer(width),
            nn.ReLU(True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        size = feats[-1].size()[2:]
        feats[-2] = F.interpolate(feats[-2], size, mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size, mode='bilinear', align_corners=True)
        feats[-4] = F.interpolate(feats[-4], size, mode='bilinear', align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)],
                         dim=1)
        return feat
class _SOLOHead123(nn.Module):
    def __init__(self,num_classes,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 sigma=0.2,
                 ins_out_channels=128,
                 ):
        super(_SOLOHead,self).__init__()
        self.num_classes = num_classes

        self.cate_out_channels = self.num_classes - 1
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.cate_down_pos = 0
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.seg_feat_channels = seg_feat_channels
        self.seg_out_channels = ins_out_channels
        self.ins_out_channels = ins_out_channels
        self.kernel_out_channels = self.ins_out_channels * 1 * 1

        self._init_layers()
        #self.init_weight()

    def _init_layers(self):
        self.kernel_convs = nn.ModuleList()
        self.cate_convs = nn.ModuleList()
        norm = nn.BatchNorm2d  # don't freeze
        for i in range(self.stacked_convs):
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(nn.Sequential(
                nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                norm(self.seg_feat_channels),
                nn.ReLU(True)
            ))
            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                nn.Conv2d(chn, self.seg_feat_channels, 3, 1, 1, bias=False),
                norm(self.seg_feat_channels),
                nn.ReLU(True)
            ))

        self.solo_kernel = nn.Sequential(nn.Conv2d(self.seg_feat_channels, self.kernel_out_channels, 1, padding=0))
        self.solo_cate = nn.Conv2d(self.seg_feat_channels, self.cate_out_channels, 3, padding=1)


    def init_weight(self):
        for m in self.mask_convs:
            torch.nn.init.normal_(m[0].weight, std=0.01)
            torch.nn.init.constant_(m[0].bias, 0)

        for m in self.kernel_convs:
            torch.nn.init.normal_(m[0].weight, std=0.01)
            torch.nn.init.constant_(m[0].bias, 0)

        for m in self.cate_convs:
            torch.nn.init.normal_(m[0].weight, std=0.01)
            torch.nn.init.constant_(m[0].bias, 0)

        prior_prob = 0.01 #self.cfg.MODEL.SOLOV2.PRIOR_PROB
        bias_init = float(-math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.normal_(self.solo_cate.weight, std=0.01)
        torch.nn.init.constant_(self.solo_cate.bias, bias_init)

    def get_coord_feat(self,b,w,h):
        x_range = torch.linspace(-1, 1,w, device='cuda')
        y_range = torch.linspace(-1, 1, h, device='cuda')
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward(self, feats, feature_pred):

        kernel_preds=[]
        cate_preds =[]
        for i,k in enumerate(['p2']):
            # kernel branch
            feat=feats[k]
            coord_feat=self.get_coord_feat(b=feat.shape[0],w=feat.shape[2],h=feat.shape[3])
            kernel_feat=torch.cat([feat,coord_feat],1)
            for i, kernel_layer in enumerate(self.kernel_convs):
                kernel_feat = kernel_layer(kernel_feat)
            kernel_pred = self.solo_kernel(kernel_feat)

            # cate branch
            cate_feat=feat
            for i, cate_layer in enumerate(self.cate_convs):
                cate_feat = cate_layer(cate_feat)
            cate_pred = self.solo_cate(cate_feat)
            kernel_preds.append(kernel_pred)
            cate_preds.append(cate_pred)

        return feature_pred, kernel_preds, cate_preds

    def losses_pseudo(self, feature_preds, kernel_preds, cate_preds, gt_ins, gt_cates):
        loss_ins=[]
        loss_cate=[]
        loss_maskiou=[]
        device = 'cuda'
        N, _, h, w = feature_preds.shape

        for batch_idx in range(N):

            feature_pred=feature_preds[batch_idx]
            kernel_pred=kernel_preds[batch_idx]
            cate_pred=cate_preds[batch_idx]

            gt_labels_raw = gt_cates[batch_idx]# gt_instances['labels'][batch_idx]
            gt_masks_raw = gt_ins[batch_idx]# gt_instances['ins_masks'][batch_idx]

            grid_size = h // 2 ** 2
            cate_label_np = np.zeros([self.num_class - 1, grid_size, grid_size], dtype=np.float)
            ins_label = torch.zeros([grid_size ** 2, w, h], dtype=torch.int16, device=device)
            ins_ind_label = torch.zeros([grid_size ** 2], dtype=torch.bool, device=device)

            if gt_masks_raw is not None:
                gt_labels = gt_labels_raw
                gt_masks = gt_masks_raw.cpu().numpy()

                for seg_mask, gt_label in zip(gt_masks, gt_labels):
                    center_w, center_h, width, height = get_ins_info(seg_mask,method='bbox')
                    coord_h = int((center_h / h) / (1. / grid_size))
                    coord_w = int((center_w / w) / (1. / grid_size))
                    radius = max(gaussian_radius((width, height), 0.3), 0)
                    with torch.no_grad():
                        pseudo_label=int(torch.argmax(cate_pred[:, coord_w, coord_h].sigmoid()).cpu().numpy())
                    temp = draw_gaussian(cate_label_np[pseudo_label], (coord_w, coord_h), (radius / 4))
                    temp = torch.from_numpy(temp)
                    non_zeros=(temp>0.5).nonzero(as_tuple=True)
                    seg_mask = torch.Tensor(seg_mask).short().cuda()
                    label = non_zeros[0] * grid_size + non_zeros[1]  #int(coord_h * grid_size + coord_w)#
                    ins_label[label, :, :] = seg_mask
                    ins_ind_label[label] = True
                kernel_pred = kernel_pred.permute(1, 2, 0).contiguous().view(-1, self.ins_out_channels)
                kernel_pred = torch.cat([kernel_pred[ins_ind_label]], 0).view(-1, self.ins_out_channels).unsqueeze(-1).unsqueeze(-1)

                #kernel_pred=kernel_pred.view(-1,c//9,3,3)

                ins_pred = F.conv2d(feature_pred.unsqueeze(0), kernel_pred, stride=1).view(-1, h, w)
                ins_label = torch.cat([ins_label[ins_ind_label]], 0)
                loss_ins.append(self.ins_loss(ins_pred, ins_label))

                cate_label = torch.from_numpy(cate_label_np).cuda().float()
                loss_cate.append(self.local_focal(cate_pred.sigmoid(), cate_label))
            else:
                continue
        return {
            'loss_ins': torch.stack(loss_ins).mean(),
            'loss_cate':  torch.stack(loss_cate).mean(),
            'loss_maskiou':torch.stack(loss_maskiou).mean() if self.mask_rescoring else 0
        }



    def seg_updata_CutMix(self, train_data):
        image = train_data['image']

        lam = np.random.beta(1.0,1.0)

        rand_index = torch.randperm(image.size()[0]).cuda().cpu().numpy().tolist()

        target_cate_a=train_data['labels']

        target_ins_a=train_data['ins_masks']

        target_cate_b=self.re_index(train_data['labels'],rand_index)#train_data['labels'][rand_index]

        target_ins_b=self.re_index(train_data['ins_masks'],rand_index)#train_data['ins_masks'][rand_index]

        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image.size(), lam)

        image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
        # compute output

        self.opt.zero_grad()

        feature_preds, kernel_preds, cate_preds = self.model(image)

        losses_a = self.losses(feature_preds, kernel_preds, cate_preds, target_ins_a,target_cate_a)

        losses_b = self.losses(feature_preds, kernel_preds, cate_preds, target_ins_b, target_cate_b)

        losses_ins = (losses_a['loss_ins'] * lam + losses_b['loss_ins']* (1. - lam)) * self.lambda_ins

        losses_cate = (losses_a['loss_cate'] * lam + losses_b['loss_cate']* (1. - lam)) * self.lambda_cate

        self.loss_seg_total = losses_ins.mean()+losses_cate.mean()

        self.loss_seg_total.backward()

        self.opt.step()

        return losses_ins.mean().item(),losses_cate.mean().item() ,0

    def rand_bbox(self,size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def get_MaskIou(self,pred,gt):
        dim0= gt.shape[0]
        smooth=1e-6
        pred=pred.contiguous().view(dim0, -1).long()
        gt=gt.contiguous().view(dim0, -1).long()
        inter=torch.sum(torch.logical_and(pred,gt), dim=1)+smooth
        union=torch.sum(torch.logical_or(pred,gt), dim=1)+smooth
        return inter/union

    def valid_cate(self, valid_data):
        image = valid_data['image']

        feature_preds, kernel_preds, cate_preds = self.model(image)

        gt_instances=valid_data
        results={'tn':[],'fp':[],'fn':[],'tp':[],'l1':[]}
        N, c, h, w = feature_preds.shape
        for batch_idx in range(N):
            cate_pred = cate_preds[batch_idx]
            gt_labels_raw = gt_instances['labels'][batch_idx]

            grid_size = h // 2 ** 2
            gt_masks_raw = gt_instances['ins_masks'][batch_idx]
            #cate_label = np.zeros([grid_size, grid_size], dtype=np.int64)
            cate_label_np = np.zeros([self.num_class - 1, grid_size, grid_size], dtype=np.float)
            if gt_masks_raw is not None:
                gt_labels = gt_labels_raw
                gt_masks = gt_masks_raw.cpu().numpy().astype(np.uint8)

                for seg_mask, gt_label in zip(gt_masks, gt_labels):
                    #center_h, center_w = ndimage.measurements.center_of_mass(seg_mask)
                    center_w, center_h, width, height = get_ins_info(seg_mask, method='bbox')
                    coord_w = int((center_w / w) / (1. / grid_size))
                    coord_h = int((center_h / h) / (1. / grid_size))
                    radius = max(gaussian_radius((width, height), 0.3), 0)
                    temp = draw_gaussian(cate_label_np[gt_label - 1], (coord_w, coord_h), radius / 4)
            results['l1']= np.mean(np.abs((cate_pred.cpu().numpy())-cate_label_np))
            #cate_pred=(cate_pred.sigmoid()>0.4).cpu().numpy().astype(np.int64)
            #cate_label = torch.from_numpy(cate_label_np).cuda()
            ##tn, fp, fn, tp= metrics.confusion_matrix(y_true=cate_pred.flatten(), y_pred=cate_label.flatten()).ravel()
            #results['tn'].append(tn)
            #results['fp'].append(fp)
            #results['fn'].append(fn)
            #results['tp'].append(tp)

        return results

    def __getitem__back(self, index):
        img = self.images[index]
        mask = self.masks[index]
        label_dic = self.labels[index]

        shape_augs = self.shape_augs.to_deterministic()
        img = shape_augs.augment_image(img)
        masks = shape_augs.augment_image(mask)

        input_augs = self.input_augs.to_deterministic()
        img = input_augs.augment_image(img)

        labels=[]
        ins_masks=[]

        for i, ui in enumerate(np.unique(masks)):
            if ui ==0:
                assert i==ui
                continue

            tmp_mask=masks==ui
            label=label_dic[ui]
            ins_masks.append(((tmp_mask)*1).astype(np.uint8))
            labels.append(label)



        if len(labels)>0:
            labels=torch.from_numpy(np.array(labels)).long()
            ins_masks=torch.from_numpy(np.stack(ins_masks,0))

        else:
            labels=None
            ins_masks=None

        image=self.transform(img)
        output={'image': image, 'labels':labels, 'ins_masks':ins_masks}
        return output
    def seg_updata_back(self, train_data):
        self.opt.zero_grad()

        image = train_data['image']
        target_ins, target_cate=train_data['ins_masks'],train_data['labels']
        if self.use_mixed:
            with autocast():
                feature_preds, kernel_preds, cate_preds = self.model(image)
                losses = self.losses(feature_preds, kernel_preds, cate_preds, target_ins, target_cate)
            self.loss_seg_total = losses['loss_ins'] * self.lambda_ins + losses['loss_cate'] * self.lambda_cate
            self.scaler.scale(self.loss_seg_total).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            feature_preds, kernel_preds, cate_preds = self.model(image)
            losses = self.losses(feature_preds, kernel_preds, cate_preds, target_ins, target_cate)
            self.loss_seg_total = losses['loss_ins'] * self.lambda_ins + losses['loss_cate'] * self.lambda_cate
            self.loss_seg_total.backward()
            self.opt.step()
        return losses['loss_ins'].item() * self.lambda_ins, losses['loss_cate'].item() * self.lambda_cate, losses['loss_maskiou'].item() if self.mask_rescoring else 0
    def losses_back(self, feature_preds, kernel_preds, cate_preds, gt_ins,gt_cates):
        loss_ins=[]
        loss_cate=[]
        loss_maskiou=[]
        device = 'cuda'
        N, _, h, w = feature_preds.shape

        for batch_idx in range(N):

            feature_pred=feature_preds[batch_idx]
            kernel_pred=kernel_preds[batch_idx]
            cate_pred=cate_preds[batch_idx]

            gt_labels_raw = gt_cates[batch_idx]
            gt_masks_raw = gt_ins[batch_idx]

            grid_size = h // 2 ** 2
            cate_label_np = np.zeros([self.num_class - 1, grid_size, grid_size], dtype=np.float)
            ins_label = torch.zeros([grid_size ** 2, w, h], dtype=torch.int16, device=device)
            ins_ind_label = torch.zeros([grid_size ** 2], dtype=torch.bool, device=device)

            if gt_masks_raw is not None:
                gt_labels = gt_labels_raw
                gt_masks = gt_masks_raw.cpu().numpy()

                for seg_mask, gt_label in zip(gt_masks, gt_labels):
                    center_w, center_h, width, height = get_ins_info(seg_mask,method='bbox')
                    radius = max(gaussian_radius((width, height), 0.3), 0)
                    coord_h = int((center_h / h) / (1. / grid_size))
                    coord_w = int((center_w / w) / (1. / grid_size))
                    temp = draw_gaussian(cate_label_np[gt_label-1], (coord_w, coord_h), (radius / 4))
                    temp = torch.from_numpy(temp)
                    non_zeros=(temp>0.5).nonzero(as_tuple=True)
                    seg_mask = torch.Tensor(seg_mask).short().cuda()
                    label = non_zeros[0] * grid_size + non_zeros[1] #label = int(coord_h * grid_size + coord_w)#
                    ins_label[label, :, :] = seg_mask
                    ins_ind_label[label] = True

                kernel_pred = kernel_pred.permute(1, 2, 0).contiguous().view(-1, self.ins_out_channels * self.kernel_size * self.kernel_size)
                kernel_pred = torch.cat([kernel_pred[ins_ind_label]], 0).view(-1, self.ins_out_channels, self.kernel_size, self.kernel_size)
                ins_pred = F.conv2d(feature_pred.unsqueeze(0), kernel_pred, stride=1).view(-1, h, w)
                ins_label = torch.cat([ins_label[ins_ind_label]], 0)
                loss_ins.append(self.ins_loss(ins_pred, ins_label))

                cate_label = torch.from_numpy(cate_label_np).cuda().float()
                loss_cate.append(self.local_focal(cate_pred.sigmoid(), cate_label))
            else:
                continue
        return {
            'loss_ins': torch.stack(loss_ins).mean(),
            'loss_cate':  torch.stack(loss_cate).mean(),
            'loss_maskiou':torch.stack(loss_maskiou).mean() if self.mask_rescoring else 0
        }