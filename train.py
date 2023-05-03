import os,shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from utils.dataloader import NucleiDataset
from trainer import Trainer
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
from utils import prepare_sub_folder,get_config,collate_func
import torch
import numpy as np
import math
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/kumar_notype_large.yaml')
parser.add_argument('--name', type=str, default='tmp')
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
    train_dataset=NucleiDataset(config,opts.seed,is_train=True)
    check_manual_seed(opts.seed)
    train_loader=DataLoader(dataset=train_dataset, batch_size=config['train']['batch_size'], shuffle=True, drop_last=True, num_workers=config['train']['num_workers'],collate_fn=collate_func,pin_memory=True)

    output_directory = os.path.join(opts.output_dir, opts.name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config,os.path.join(output_directory,'config.yaml'))

    trainer = Trainer(config)
    trainer.cuda()

    iteration=0
    iter_per_epoch=len(train_loader)

    for epoch in range(config['train']['max_epoch']):
        for train_data in train_loader:
            for k in train_data.keys():
                if not isinstance(train_data[k], list):
                    train_data[k] = train_data[k].cuda().detach()
                else:
                    train_data[k] = [s.cuda().detach() if s is not None else s for s in train_data[k]]

            ins_loss, cate_loss, maskiou_loss = trainer.seg_updata_FMIX(train_data)

            sys.stdout.write(
                f'\r epoch:{epoch} step:{iteration}/{iter_per_epoch} ins_loss: {ins_loss} cate_loss: {cate_loss} maskiou_loss: {maskiou_loss}')
            iteration += 1
        if (epoch + 1) % 20 == 0:
            trainer.save(checkpoint_directory, epoch)
        trainer.scheduler.step()

    trainer.save(checkpoint_directory, epoch)

