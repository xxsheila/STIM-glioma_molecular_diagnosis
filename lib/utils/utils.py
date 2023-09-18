

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.2):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



def create_logger(cfg, cfg_name, phase='train'):
    """生成logger文件

    Args:
        cfg ([type]): [description]
        cfg_name ([type]): [description]
        phase (str, optional): [description]. Defaults to 'train'.

    Returns:
        [type]: [description]
    """
    root_output_dir = Path(cfg["OUTPUT_DIR"])
    # root_output_dir = Path('output/')
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    dataset = cfg['DATASET']['DATASET']
    model = cfg['MODEL']['NAME']
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    if phase=='test':
        final_output_dir = root_output_dir / "test"/ dataset / model
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
        return logger, log_file
    
    else:
        final_output_dir = root_output_dir / dataset / model / cfg_name

        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
        final_log_file = final_output_dir / log_file
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=str(final_log_file),
                            format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        tensorboard_log_dir = Path(cfg['LOG_DIR']) / dataset / model / (cfg_name + '_' + time_str)
        print('=> creating {}'.format(tensorboard_log_dir))
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        return logger, str(final_output_dir), str(tensorboard_log_dir), log_file




def get_optimizer(cfg, model):
    """获取优化器

    Args:
        cfg ([type]): [description]
        model ([type]): [description]

    Returns:
        [type]: [description]
    """
    optimizer = None
    if cfg['TRAIN']['OPTIMIZER'] == 'sgd':
        optimizer = optim.SGD(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['TRAIN']['LR'],
            momentum=cfg['TRAIN']['MOMENTUM'],
            weight_decay=cfg['TRAIN']['WD'],
            nesterov=cfg['TRAIN']['NESTEROV']
        )
    elif cfg['TRAIN']['OPTIMIZER'] == 'adam':
        optimizer = optim.Adam(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['TRAIN']['LR'],
            weight_decay=cfg['TRAIN']['WD'],
        )
    elif cfg['TRAIN']['OPTIMIZER'] == 'rmsprop':
        optimizer = optim.RMSprop(
            #model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['TRAIN']['LR'],
            momentum=cfg['TRAIN']['MOMENTUM'],
            weight_decay=cfg['TRAIN']['WD'],
            alpha=cfg['TRAIN']['RMSPROP_ALPHA'],
            centered=cfg['TRAIN']['RMSPROP_CENTERED']
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    """模型保存

    Args:
        states ([type]): [description]
        is_best (bool): [description]
        output_dir ([type]): [description]
        filename (str, optional): [description]. Defaults to 'checkpoint.pth.tar'.
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


###################################################################################
# resnet 加载

# def load_partical_parameter(current_model, param_path=None):
#     """加载部分参数

#     Args:
#         current_model (模型): 要加载参数的模型
#         param_path ([str], optional): 保存的模型参数的路径. Defaults to None.
#     """
#     param = torch.load(param_path)  # 模型加载的参数
#     # for k,v in param.items():
#     #     print(k)
#     param.pop('fc.weight')
#     param.pop('fc.bias')
#     current_model.resnet.load_state_dict(param)

# sppnet中的resnet加载

# def param_filtrate(old_param, layer_wanted=None):
#     """参数筛选

#     Args:
#         old_param (OrderedDict): 传入的有待筛选的参数
#         layer_wanted (str, optional): 要选用的网络参数的前缀名. Defaults to None.

#     Returns:
#         OrderedDict: 筛选出来的参数
#     """
#     new_param = OrderedDict()
#     for k, v in old_param.items():
#         # if k.startswith(layer_wanted) and k not in ["resnet.fc.weight", "resnet.fc.bias", "resnet.resnet.fc.weight", "resnet.resnet.fc.bias"] and not k.startswith('glore2d'):
#         if k not in ["fc.weight", "fc.bias", "fc.weight", "fc.bias"]:
#             new_param[k] = v
#     return new_param


# def load_partical_parameter(current_model, param_path=None):
#     """加载部分参数

#     Args:
#         current_model (模型): 要加载参数的模型
#         param_path ([str], optional): 保存的模型参数的路径. Defaults to None.
#     """
#     param = torch.load(param_path)  # 模型加载的参数
#     param = param_filtrate(param)  # 参数筛选


#     model_dict =  current_model.resnet.state_dict()
#     state_dict = {k:v for k,v in param.items() if k in model_dict.keys()}
#     # print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
#     model_dict.update(state_dict)
#     current_model.resnet.load_state_dict(model_dict)