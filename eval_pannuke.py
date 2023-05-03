import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='pannuke_exp')
parser.add_argument('--save_path', type=str, default='pannuke_outputs')
parser.add_argument('--train_fold', type=int, default=1)
parser.add_argument('--test_fold', type=int, default=3)
opts = parser.parse_args()

pred_path = rf'outputs/{opts.name}/train_{opts.train_fold}_to_test_{opts.test_fold}'
true_path = rf'datasets/PanNuKe/masks/fold{opts.test_fold}'
save_path = opts.save_path
cmd=rf"python PanNuke-metrics/run.py --true_path={true_path} --pred_path={pred_path} --save_path={save_path}"

os.system(cmd)
