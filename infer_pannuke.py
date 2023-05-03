from trainer import Trainer
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import scipy.io as scio
from utils.dataloader import NucleiDataset,PannukeDataset
import torch
import os
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils import get_config,collate_func
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--name', type=str, default='pannuke227')
parser.add_argument('--train_fold', type=int, default=2)
parser.add_argument('--test_fold', type=int, default=3)
parser.add_argument('--epoch',type=int,default=100)
opts = parser.parse_args()

def stack_prediction(seg_masks,cate_labels):
    out_seg=np.zeros((256,256,6))
    idx_num=1
    for mask,label in zip(seg_masks,cate_labels):
        assert label!=5
        out_seg[:,:,label]=np.maximum(out_seg[:,:,label],mask*idx_num)
        idx_num+=1
    out_seg[:,:,5]=np.sum(out_seg[:,:,:5],axis=-1)==0
    return out_seg

if __name__ == '__main__':
    opts.config=os.path.join(opts.output_dir,'{}'.format(opts.name),'train_{}_to_test_{}/config.yaml'.format( opts.train_fold,opts.test_fold))
    config=get_config(opts.config)

    #train_dataset=NucleiDataset(data_root=config['dataroot'],is_train=True,stain_norm=stain_norm_type)
    test_dataset=PannukeDataset(data_root=config['dataroot'], is_train=False, fold=opts.test_fold,output_stride=config['model']['output_stride'])
    test_loader=DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,collate_fn=collate_func,pin_memory=True)
    config['model']['kernel_size']=1
    config['train']['use_mixed']=False
    trainer = Trainer(config)
    trainer.cuda()

    state_path = os.path.join(opts.output_dir,opts.name,'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold),'checkpoints/model_{}.pt'.format('%04d' % (opts.epoch)))
    #state_path = os.path.join(opts.output_dir,opts.name,'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold),'checkpoints/model_ema.pt')
    state_dict = torch.load(state_path)

    trainer.model.load_state_dict(state_dict['seg'])
    predictions=[]
    for test_data in test_loader:
        for k in test_data.keys():
            if not isinstance(test_data[k], list):
                test_data[k] = test_data[k].cuda().detach()
            else:
                test_data[k] = [s.cuda().detach() if s is not None else s for s in test_data[k]]
        with torch.no_grad():
            img=test_data['image']
            output = trainer.prediction(img, score_thr=0.4, update_thr=0.2)
        if output is not None:
            seg_masks, cate_labels, cate_scores = output
            seg_masks = seg_masks.cpu().numpy()
            cate_labels = cate_labels.cpu().numpy()
            cate_scores = cate_scores.cpu().numpy()
            predictions.append(stack_prediction(seg_masks, cate_labels))
        else:
            predictions.append(np.zeros((256,256,6)))

    predictions=np.stack(predictions,0).astype(np.int16)
    save_fp= os.path.join(opts.output_dir,opts.name,'train_{}_to_test_{}/masks.npy'.format( opts.train_fold,opts.test_fold))

    np.save(save_fp,predictions)




