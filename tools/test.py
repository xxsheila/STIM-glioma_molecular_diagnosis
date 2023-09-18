# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# Modified by BgZhang
# Modified by XXie
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import ptvsd
# ptvsd.enable_attach(address=('10.154.63.78',5679))
# ptvsd.wait_for_attach()

import argparse
import os
import sys
import shutil
import pprint


import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path


import _init_paths
import models
from config import read_config
from core.function import test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
from utils.my_dataset import My_Dataset, MyROI_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="./experiments/spp_glore_pairwise.json",
                        required=True,
                        type=str)

    args = parser.parse_args()
    cfg = read_config(args)

    return cfg, args


def main():
    config, args = parse_args()

    logger, log_file_name = create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config['CUDNN']['BENCHMARK']
    torch.backends.cudnn.deterministic = config['CUDNN']['DETERMINISTIC']
    torch.backends.cudnn.enabled = config['CUDNN']['ENABLED']

    # TODO 这里需要和train.py文件中的相同
    kwargs = {"num_classes":2}
    model = eval('models.' + 'resnet_attention_lstm.get_model')(**kwargs)

    dir_path =  Path(config["OUTPUT_DIR"]) / config['DATASET']['DATASET'] / config['MODEL']['NAME']
    model_dir_list = [p.path for p in os.scandir(dir_path)]
    for fold_i in range(len(config['DATASET']['TRAIN_SET'])+len(config['DATASET']['VAL_SET'])):
        #分别读取十折交叉训练的最优模型
        model_save_dir = model_dir_list[fold_i]
        if config['TEST']['MODEL_FILE']:
            logger.info('=> loading model from {}'.format(config['TEST']['MODEL_FILE']))
            model.load_state_dict(torch.load(config['TEST']['MODEL_FILE']))
        else:
            model_state_file = os.path.join(model_save_dir,
                                            'model_best.pth.tar')
            logger.info('=> loading model from {}'.format(model_state_file))
            model.load_state_dict(torch.load(model_state_file))

        gpus = list(config['GPUS'])
        # model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        model.cuda()

        # define loss function (criterion) and optimizer
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2).cuda()

        # Data loading code
        for i in range(len(config['TEST']["ROOT"])):
            valdir = config['TEST']["ROOT"][i]
            logger.info(valdir)


            valid_loader = torch.utils.data.DataLoader(
            My_Dataset(
                img_dir_path=valdir, 
                forder_list=config['DATASET']['TEST_SET'],
                json_dir=config['DATASET']['JSONDIR'],
                patch_class=config['DATASET']['PATCH_CLASS'],
                model_input_size=config["MODEL"]["IMAGE_SIZE"],
                threshold_norm=config["DATASET"]["THRESHOLD_NORM"],
                patch_view=config["TEST"]["PATCH_VIEW"],
                mode='val'
            ),
                batch_size=config['TEST']['BATCH_SIZE_PER_GPU']*len(gpus),
                shuffle=False,
                num_workers=config['WORKERS'],
                pin_memory=True
            )
            # evaluate on validation set
            output_all = test(config, valid_loader, model, criterion, None)

if __name__ == '__main__':
    main()