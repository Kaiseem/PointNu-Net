from utils.dataloader import NucleiDataset,PannukeDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import sys
from utils import prepare_sub_folder,get_config,collate_func
import torch
import numpy as np
import os,shutil
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/pannuke.yaml')
parser.add_argument('--name', type=str, default='pannuke_experiment')
parser.add_argument('--train_fold', type=int, default=2)
parser.add_argument('--val_fold', type=int, default=1)
parser.add_argument('--test_fold', type=int, default=3)
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--seed', type=int, default=10)
opts = parser.parse_args()

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a
    random one and communicate it to the user.
    Args:
        seed: seed to check
    """
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # ia.random.seed(seed)

    print("Using manual seed: {seed}".format(seed=seed))
    return

if __name__ == '__main__':
    config=get_config(opts.config)
    check_manual_seed(opts.seed)
    train_dataset=PannukeDataset(data_root=config['dataroot'], seed=opts.seed, is_train=True, fold=opts.train_fold,output_stride=config['model']['output_stride'])
    train_loader=DataLoader(dataset=train_dataset, batch_size=config['train']['batch_size'], shuffle=True, drop_last=True, num_workers=config['train']['num_workers'],persistent_workers=True,collate_fn=collate_func,pin_memory=True)

    output_directory = os.path.join(opts.output_dir, opts.name, 'train_{}_to_test_{}'.format( opts.train_fold,opts.test_fold))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config,os.path.join(output_directory,'config.yaml'))

    trainer = Trainer(config)
    trainer.cuda()

    iteration=0
    iter_per_epoch=len(train_loader)
    for epoch in range(config['train']['max_epoch']):
        for train_data in train_loader:
            for k in train_data.keys():
                if not isinstance(train_data[k],list):
                    train_data[k]=train_data[k].cuda().detach()
                else:
                    train_data[k] = [s.cuda().detach() if s is not None else s for s in train_data[k]]
            ins_loss, cate_loss,maskiou_loss=trainer.seg_updata(train_data)
            sys.stdout.write(f'\r epoch:{epoch} step:{iteration}/{iter_per_epoch} ins_loss: {ins_loss} cate_loss: {cate_loss} maskiou_loss: {maskiou_loss}')
            iteration+=1
        trainer.scheduler.step()

        if (epoch+1)%50==0:
            trainer.save(checkpoint_directory, epoch)




