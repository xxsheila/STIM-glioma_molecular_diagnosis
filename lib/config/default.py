import os
import json

def get_cfg():
    """
        生成config文件
    """

    cfg = {}
    cfg['GPUS'] = [0]

    cfg['LOG_DIR'] = 'log/'  # 日志
    cfg['OUTPUT_DIR'] = 'output/'  # 模型等文件路径
    cfg['WORKERS'] = 4  # 读取数据时的线程数

    cfg['MODEL'] = {}
    cfg['MODEL']['NAME'] = 'resnet'  # 模型的名字,用于创建文件夹, 一定要与模型.py名字一样
    cfg['MODEL']['IMAGE_SIZE'] = [96, 96]  # 输入网络的图片大小

    cfg['CUDNN'] = {}
    cfg['CUDNN']['BENCHMARK'] = True  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    cfg['CUDNN']['DETERMINISTIC'] = False  # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    cfg['CUDNN']['ENABLED'] = True  # cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用,如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法,然后再设置:torch.backends.cudnn.benchmark = true,那么cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题




    cfg['DATASET'] = {}
    cfg['DATASET']['DATASET'] = 'cancer_tenfolder'  # 数据集的名字
    cfg['DATASET']['ROOT'] = '/storage/xiexuan/copy1_dataset_202202281530/proposed_data'  # 数据集的路径
    cfg['DATASET']['JSONDIR'] = "jsonfile"
    cfg['DATASET']['TRAIN_SET'] = ["d_1", "d_2", "d_3", "d_4", "d_5", "d_6", "d_7", "d_8", "d_9"]  # 训练集的文件夹名字
    cfg['DATASET']['VAL_SET'] = ["d_0"]  # val的文件夹名字
    cfg['DATASET']['EVAL_SET'] = ["d_0"]  # eval的文件夹名字
    cfg['DATASET']['PATCH_CLASS'] = {"feral" : 0, "mutational" : 1, "bg":2}
    cfg['DATASET']['THRESHOLD_NORM'] = 4000.0


    cfg['TRAIN'] = {}
    cfg['TRAIN']['BATCH_SIZE_PER_GPU'] = 4
    cfg['TRAIN']['BEGIN_EPOCH'] = 0
    cfg['TRAIN']['END_EPOCH'] = 1000
    cfg['TRAIN']['RESUME'] = True

    cfg['TRAIN']['LR_SCHEDULER'] = {}
    cfg['TRAIN']['LR_SCHEDULER']['LR_SCHEDULER'] = 'EARLYSTOPPING'  # 选择lr下降策略,eg:'MultiStepLR', 'StepLR', 'ExponentialLR', 'EARLYSTOPPING'

    cfg['TRAIN']['LR_SCHEDULER']['MultiStepLR'] = {}
    cfg['TRAIN']['LR_SCHEDULER']['MultiStepLR']['LR_FACTOR'] = 0.1
    cfg['TRAIN']['LR_SCHEDULER']['MultiStepLR']['LR_STEP'] = [7, 10, 12]  # 那个epoch下降LR_FACTOR倍数

    cfg['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING'] = {}
    cfg['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['PATIENCE'] = 300  # 几次平稳后调低学习率
    cfg['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['DELTA'] = -0.0001
    cfg['TRAIN']['LR_SCHEDULER']['EARLYSTOPPING']['MIN_LR'] = 1e-5  # 最小学习率,低于这个学习率停止训练

    cfg['TRAIN']['LR_SCHEDULER']['ExponentialLR'] = {}
    cfg['TRAIN']['LR_SCHEDULER']['ExponentialLR']['EXPONENTIAL_LR_GAMMA'] = 0.9  # 指数下降方式

    cfg['TRAIN']['OPTIMIZER'] = 'adam'  # 优化器选择
    cfg['TRAIN']['LR'] = 3e-4  # 学习率
    cfg['TRAIN']['WD'] = 0.001  # 惩罚项
    cfg['TRAIN']['MOMENTUM'] = 0.9
    cfg['TRAIN']['NESTEROV'] = True
    cfg['TRAIN']['SHUFFLE'] = True
    cfg['TRAIN']['SEED'] = 3
    cfg['TRAIN']['PATCH_VIEW'] = 32

    cfg['TEST'] = {}
    cfg['TEST']['ROOT'] = [
            "/storage/xiexuan/copy1_dataset_202202281530/proposed_data"
    ]  # 测试集的路径
    cfg['TEST']['EVAL_SET'] = [
            "d_0"
    ]  # 测试集的文件夹
    cfg['TEST']['BATCH_SIZE_PER_GPU'] = 1
    cfg['TEST']['MODEL_FILE'] = ''
    cfg['TEST']['PATCH_VIEW'] = 160

    cfg['DEBUG'] = False

    return cfg


def read_config(args):
    """该函数的目的：添加args中新增的项，即该项在配置文件中没有，仅在train.py文件中的parse_args函数中新增

    Args:
        args ： 配置信息

    Returns:
        cfg : 所有的配置信息
    """
    with open(args.cfg, 'r') as f:
        cfg = json.load(f)
    
    for arg in vars(args):
        # if arg == 'cfg':
            # # 表示去除args中的某一项，这里表示去除cfg项
        #     continue
        cfg[arg] = getattr(args, arg)  # 将parse_args函数中新增的项添加到配置文件中，方便以后的记录

    return cfg



if __name__ == '__main__':
    cfg = get_cfg()
    NUM_FOLDER = 10
    PREDIX = 'd_'
    for th in [4000.0]:
        cfg['DATASET']['THRESHOLD_NORM'] = th
        for i in range(NUM_FOLDER):
            train_folder = []
            val_folder = []
            
            for j in range(NUM_FOLDER):
                if i == j:
                    val_folder.append(PREDIX + str(j))
                else:
                    train_folder.append(PREDIX + str(j))
            eval_folder = val_folder
            cfg['DATASET']['TRAIN_SET'] = train_folder
            cfg['DATASET']['VAL_SET'] = val_folder
            cfg['DATASET']['EVAL_SET'] = val_folder

            jsonfile_name = "net_" + PREDIX + str(i) + "_" + str(int(th)) + ".json"
        
            with open(os.path.join('experiments', jsonfile_name), 'w') as f:
                json.dump(cfg, f)