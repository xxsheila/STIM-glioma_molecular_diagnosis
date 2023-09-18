# 冻结一些层

def freeze_model(model, is_freeze:bool):
    """冻结model中的参数

    Args:
        model ([type]): 指定模型
        is_freeze (bool): 是否冻结模型, True表示冻结, False表示不冻结
    """
    requires_grad = False if is_freeze else True
    for p in model.parameters():
        p.requires_grad = requires_grad

def freeze_bn(m):
    """冻结bn层

    Args:
        m ([type]): 指定模型,使用:model.apply(freeze_bn)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def activate_bn(m):
    """活化bn层

    Args:
        m ([type]): 指定模型,使用:model.apply(freeze_bn)
    """
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()





# 例子, 在function中添加这些代码,实现部分层的冻结

# def train(config, train_loader, model, criterion, pdist, optimizer, epoch,
#           output_dir, tb_log_dir, writer_dict):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()

#     # switch to train mode

#     # Plan A
#     # 小于冻结epoch时,只训练自己的模块
#     if epoch < config["TRAIN"]["FREEZE_EPOCH"]:
#         freeze_model(model, is_freeze=True)  # 全部模型冻结
#         model.apply(freeze_bn)  # 全部bn冻结

#         # config["TRAIN"]["UNFREEZE_LAYER"]中存的是模型中非冻结层的变量名
#         for layer in config["TRAIN"]["UNFREEZE_LAYER"]:
#             freeze_model(
#                 eval("model." + layer), 
#                 is_freeze=False
#             )
#             eval("model." + layer).apply(activate_bn)
#     else:
#         freeze_model(model, is_freeze=False)
#         model.train()
#     # Paln B
#     # model.train() 
#     end = time.time()
