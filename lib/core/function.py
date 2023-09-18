# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by BgZhang
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import torch
import numpy as np
from tqdm import tqdm
from core.evaluate import accuracy
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc


# from sklearn import metrics
# from sklearn.metrics import roc_auc_score, auc
# from trick_dl.freeze import freeze_model, freeze_bn, activate_bn


logger = logging.getLogger(__name__)

def train(config, train_loader, patch_number, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target, _) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        patient_num, pv, ch, wi, he = input.shape
        input = input.view(-1, ch, wi, he)
        input = input.cuda(non_blocking=True) # 加载数据非阻塞
        target = target.cuda(non_blocking=True)
        output = model(input, patch_number)

        if len(output.size()) == 1:
            output = torch.unsqueeze(output, dim=0)
        
        loss = criterion(output, target)
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # prec1 = 0
        prec1 = accuracy(output, target, (1,))[0]
        top1.update(prec1[0], patient_num)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    msg = 'Epoch: [{0}]\t' \
            'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            'Speed {speed:.1f} samples/s\t' \
            'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            'LR {learning_rate}\t' \
            'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
            'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, batch_time=batch_time,
                speed=patient_num/batch_time.val,
                data_time=data_time, 
                learning_rate= optimizer.state_dict()['param_groups'][0]['lr'],
                loss=losses, top1=top1)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer.add_scalar('train_top1', top1.val, global_steps)
        writer.add_scalar('train_lr', optimizer.state_dict()['param_groups'][0]['lr'], global_steps)
        writer_dict['train_global_steps'] = global_steps + 1


def validate(config, val_loader, model, criterion, output_dir, tb_log_dir,
                         writer_dict=None, prefix='valid'):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # for auc
        scores = []  # 用来存储模型输出后的值的softmax,然后去这个softmax的第一列作为scores
        preds = []  # 模型输出的值
        labels = []  # ground truth
        
        patch_number = 0

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target, _) in enumerate(tqdm(val_loader)):
                # compute output                
                patient_num, pv, ch, wi, he = input.shape
                patch_number = pv
                input = input.view(-1, ch, wi, he)
                input = input.cuda(non_blocking=True)


                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                output = model(input, patch_number)

                # obb, _ = output.shape

                # output = output.sum(0, keepdim=True).div(obb * 1.0)  #这里batch只能是1

                # for auc
                score = torch.nn.functional.softmax(output, dim=1)
                scores.extend(score.select(dim=1, index=1).tolist())
                labels.extend(target.tolist())


                loss = criterion(output, target)

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))
                prec1 = accuracy(output, target, (1,))[0]
                top1.update(prec1[0], patient_num)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
        fnr = 1 - tpr
        eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]  # np.nanargmin:找出最小值的索引
        roc_auc = auc(fpr, tpr)  # auc值


        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy@1 {top1.avg:.3f} \t' \
              'AUC {auc:.4f}\t'\
              'EER {eer:.4f}'.format(batch_time=batch_time, loss=losses, top1=top1, auc=roc_auc, eer=eer)

        logger.info(msg)


        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict[f'{prefix}_global_steps']
            writer.add_scalar(f'{prefix}_loss', losses.avg, global_steps)
            writer.add_scalar(f'{prefix}_top1', top1.avg, global_steps)
            writer.add_scalar(f'{prefix}_auc', roc_auc, global_steps)
            writer_dict[f'{prefix}_global_steps'] = global_steps + 1

        return top1.avg, roc_auc


def test(config, val_loader, model, criterion, writer_dict=None, prefix='valid'):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # for auc
        scores = []  # 用来存储模型输出后的值的softmax,然后去这个softmax的第一列作为scores
        preds = []  # 模型输出的值
        labels = []  # ground truth
        # output_all = [] #模型attention后的预测值
        # weight1 = [] #模块1的输出
        # weight2 = [] #模块2的输出
        names = []

        patch_number = 0

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target, name) in enumerate(tqdm(val_loader)):
                # compute output

                patient_num, pv, ch, wi, he = input.shape
                patch_number = pv
                input = input.view(-1, ch, wi, he)
                input = input.cuda(non_blocking=True)


                input = input.cuda(non_blocking=True)
                target = target.type(torch.LongTensor).cuda(non_blocking=True)
                output = model(input, patch_number)

                # for auc
                score = torch.nn.functional.softmax(output, dim=1)
                scores.extend(score.select(dim=1, index=1).tolist())
                labels.extend(target.tolist())
                preds.extend(output.max(dim=1)[1].data)
                names.extend(name)


                loss = criterion(output, target)

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))
                prec1 = accuracy(output, target, (1,))[0]
                top1.update(prec1[0], patient_num)
                # print('top1', prec1[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
        fnr = 1 - tpr
        eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]  # np.nanargmin:找出最小值的索引
        roc_auc = auc(fpr, tpr)  # auc值


        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy@1 {top1.avg:.3f} \t' \
              'AUC {auc:.4f}\t'\
              'EER {eer:.4f}\n'\
              'labels:{labels}\n'\
              'preds:{preds}\n'\
              'scores:{scores}\n'\
              'names:{names}\n'\
              'fpr:{fpr}\n'\
              'tpr:{tpr}\n'.format(batch_time=batch_time, loss=losses, top1=top1, auc=roc_auc, eer=eer, labels=labels, preds=preds, scores=scores, names=names, fpr=fpr, tpr=tpr)

        logger.info(msg)


        return labels


def test_weight(config, val_loader, model, criterion, writer_dict=None, prefix='valid'):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # for auc
        scores = []  # 用来存储模型输出后的值的softmax,然后去这个softmax的第一列作为scores
        preds = []  # 模型输出的值
        labels = []  # ground truth
        output_all = [] #模型attention后的预测值
        weight1 = [] #模块1的输出
        weight2 = [] #模块2的输出
        names = []

        patch_number = 0

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target, name) in enumerate(tqdm(val_loader)):
                # compute output

                bb, pv, ch, wi, he = input.shape
                patch_number = pv
                input = input.view(-1, ch, wi, he)
                input = input.cuda(non_blocking=True)


                input = input.cuda(non_blocking=True)
                target = target.type(torch.LongTensor).cuda(non_blocking=True)
                # output = model(input, patch_number)
                wet1= model(input, patch_number)

                # print(output, target)
                # output_all.append(output.squeeze().tolist())
                weight1.append(wet1.squeeze().tolist())
                
                # print('output',output.size()) #(1,2)
                # print('input.size0', bb)
                
                # obb, _ = output.shape

                # output = output.sum(0, keepdim=True).div(obb * 1.0)  # 这里batch只能是1

        #         # for auc
        #         score = torch.nn.functional.softmax(output, dim=1)
        #         scores.extend(score.select(dim=1, index=1).tolist())
                labels.extend(target.tolist())
        #         preds.extend(output.max(dim=1)[1].data)
        #         names.extend(name)


        #         loss = criterion(output, target)

        #         # measure accuracy and record loss
        #         losses.update(loss.item(), input.size(0))
        #         prec1 = accuracy(output, target, (1,))[0]
        #         top1.update(prec1[0], bb)
        #         # print('top1', prec1[0])

        #         # measure elapsed time
        #         batch_time.update(time.time() - end)
        #         end = time.time()


        # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, drop_intermediate=False)
        # fnr = 1 - tpr
        # eer = fnr[np.nanargmin(np.absolute((fnr - fpr)))]  # np.nanargmin:找出最小值的索引
        # roc_auc = auc(fpr, tpr)  # auc值


        # msg = 'Test: Time {batch_time.avg:.3f}\t' \
        #       'Loss {loss.avg:.4f}\t' \
        #       'Accuracy@1 {top1.avg:.3f} \t' \
        #       'AUC {auc:.4f}\t'\
        #       'EER {eer:.4f}\n'\
        #       'labels:{labels}\n'\
        #       'preds:{preds}\n'\
        #       'scores:{scores}\n'\
        #       'names:{names}\n'\
        #           'fpr:{fpr}\n'\
        #               'tpr:{tpr}\n'.format(batch_time=batch_time, loss=losses, top1=top1, auc=roc_auc, eer=eer, labels=labels, preds=preds, scores=scores, names=names, fpr=fpr, tpr=tpr)

        # logger.info(msg)


        return weight1, labels

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 表示正确率
        self.sum += val * n  # 表示正确的个数
        self.count += n
        self.avg = self.sum / self.count