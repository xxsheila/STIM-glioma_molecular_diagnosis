# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by BgZhang
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import matplotlib.pyplot as plt

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def draw_roc_curve(fpr, tpr, auc, image_path_name):
    """绘制roc曲线

    Args:
        fpr ([type]): [description]
        tpr ([type]): [description]
        auc ([type]): [description]
        image_path_name ([type]): 图片的路径和名字,形如:output/.../ROC.png
    """
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold (AUC = %0.4f)' % (auc))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(image_path_name)