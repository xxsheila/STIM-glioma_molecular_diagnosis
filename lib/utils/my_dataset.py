import os, json, random, torch
import scipy.io as sio
from torch.utils.data.dataset import Dataset
import numpy as np
from .transforms_bgzhang import get_transform
import h5py
from scipy import fftpack
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


class My_Dataset(Dataset):
    def __init__(self,
        img_dir_path:str, 
        forder_list:list,
        json_dir:str,
        patch_class:dict,
        model_input_size:list,
        threshold_norm:float,
        patch_view:int,
        mode='train'
    ):
        """训练数据集读取

        Args:
            img_dir_path (str): 数据集的路径
            forder_list (list): 训练集的文件夹
            json_dir (str): json文件的文件夹
            patch_class (dict): patch的分类
            model_input_size (list): 模型输入图像的尺寸
            threshold_norm (float): 归一化的阈值, 使得 (-threshold_norm <= 输入矩阵 <= threshold_norm)
        """

        self.img_dir_path = img_dir_path
        self.patch_class = patch_class
        self.patch_view = patch_view
        self.mode = mode
        self.json_files_path = []
        for folder_name in forder_list:
            json_files_list = [p.path for p in os.scandir(os.path.join(img_dir_path, json_dir, folder_name))]
            self.json_files_path.extend(json_files_list)

        self.my_transforms = get_transform(threshold_norm)


    def __getitem__(self, index):
        file_path = self.json_files_path[index]

        file_path_split = file_path.split('/')

        n_folder_name = file_path_split[-2]
        file_without_suffix = file_path_split[-1].split('.')[0]
        k = n_folder_name + '/' + file_without_suffix


        matfile = read_matfile(os.path.join(self.img_dir_path, file_path_split[-2], file_without_suffix + '.mat'))
        max_rf = matfile['max_rf'].astype(np.float32)
        label = matfile['IDH_patient'].astype(np.long)

        # train时随机取self.patch_view个图像块，test时取前self.patch_view个数据块
        with open(file_path, 'r') as f:
            jsonfile = json.load(f)
            patch_list = jsonfile[k]

            if self.mode == 'train':
                tmp_list = list(range(len(patch_list)))
                tmp_index = random.sample(tmp_list, self.patch_view)

                tmp_index.sort()

                selected_patch_list = [patch_list[i] for i in tmp_index]

            else:
                selected_patch_list = patch_list
                # tmp_list = list(range(len(patch_list)))
                # tmp_index = random.sample(tmp_list, self.patch_view)

                # tmp_index.sort()

                # selected_patch_list = [patch_list[i] for i in tmp_index]

            img_file_stack = []

            # 提取图像块并预处理
            for pp in selected_patch_list:

                _, x, y, patch_length, patch_width, _, _, maxv, minv = pp

                # patch_label = self.patch_class[patch_label]

                data = max_rf[x:x+patch_length, y:y+patch_width]

                img_transform = self.my_transforms((data, maxv, minv) )

                img_file_stack.append(img_transform)


            img_file_stack = torch.stack(img_file_stack, dim=0)
            return img_file_stack, np.array(label), file_without_suffix

    def __len__(self):
        return len(self.json_files_path)


class MyROI_Dataset(Dataset):
    def __init__(self,
        img_dir_path:str, 
        forder_list:list,
        json_dir:str,
        patch_class:dict,
        model_input_size:list,
        threshold_norm:float,
        patch_view:int,
        mode='train'
    ):
        """训练数据集读取

        Args:
            img_dir_path (str): 数据集的路径
            forder_list (list): 训练集的文件夹
            json_dir (str): json文件的文件夹
            patch_class (dict): patch的分类
            model_input_size (list): 模型输入图像的尺寸
            threshold_norm (float): 归一化的阈值, 使得 (-threshold_norm <= 输入矩阵 <= threshold_norm)
        """

        self.img_dir_path = img_dir_path
        self.patch_class = patch_class
        self.patch_view = patch_view
        self.mode = mode
        self.json_files_path = []
        for folder_name in forder_list:
            json_files_list = [p.path for p in os.scandir(os.path.join(img_dir_path, json_dir, folder_name))]
            self.json_files_path.extend(json_files_list)

        self.my_transforms = get_transform(threshold_norm)


    def __getitem__(self, index):
        file_path = self.json_files_path[index]

        file_path_split = file_path.split('/')

        n_folder_name = file_path_split[-2]
        file_without_suffix = file_path_split[-1].split('.')[0]
        k = n_folder_name + '/' + file_without_suffix


        matfile = read_matfile(os.path.join(self.img_dir_path, file_path_split[-2], file_without_suffix + '.mat'))
        max_rf = matfile['max_rf'].astype(np.float32)
        label = matfile['IDH_patient'].astype(np.long)

        # train时随机取self.patch_view个图像块，test时取所有数据块
        with open(file_path, 'r') as f:
            jsonfile = json.load(f)
            patch_list_all = jsonfile[k] # 所有的patch
            # print(len(patch_list_all))
            patch_list = get_ROI_patches(patch_list_all) #ROI区域的patch
            # print(len(patch_list))

            if self.mode == 'train':
                tmp_list = list(range(len(patch_list)))
                tmp_index = random.sample(tmp_list, self.patch_view)

                tmp_index.sort()

                selected_patch_list = [patch_list[i] for i in tmp_index]

            else:
                selected_patch_list = patch_list

            img_file_stack = []

            # 提取图像块并预处理
            for pp in selected_patch_list:

                _, x, y, patch_length, patch_width, _, _, maxv, minv = pp

                # patch_label = self.patch_class[patch_label]

                data = max_rf[x:x+patch_length, y:y+patch_width]

                img_transform = self.my_transforms((data, maxv, minv) )

                img_file_stack.append(img_transform)


            img_file_stack = torch.stack(img_file_stack, dim=0)
            return img_file_stack, np.array(label), file_without_suffix

    def __len__(self):
        return len(self.json_files_path)


class HeatMap_Dataset(Dataset):
    def __init__(self,
        img_dir_path:str, 
        forder_list:list,
        json_dir:str,
        patch_class:dict,
        model_input_size:list,
        threshold_norm:float,
        mode='train'
    ):
        """训练数据集读取

        Args:
            img_dir_path (str): 数据集的路径
            forder_list (list): 训练集的文件夹
            json_dir (str): json文件的文件夹
            patch_class (dict): patch的分类
            model_input_size (list): 模型输入图像的尺寸
            threshold_norm (float): 归一化的阈值, 使得 (-threshold_norm <= 输入矩阵 <= threshold_norm)
            save_fig_path(str):绘制超声RF信号对应的超声灰度图像
        """

        self.img_dir_path = img_dir_path
        self.patch_class = patch_class
        self.mode = mode
        self.json_files_path = []
        for folder_name in forder_list:
            json_files_list = [p.path for p in os.scandir(os.path.join(img_dir_path, json_dir, folder_name))]
            self.json_files_path.extend(json_files_list)

        self.my_transforms = get_transform(threshold_norm)


    def __getitem__(self, index):
        file_path = self.json_files_path[index]

        file_path_split = file_path.split('/')

        n_folder_name = file_path_split[-2]
        file_without_suffix = file_path_split[-1].split('.')[0]
        k = n_folder_name + '/' + file_without_suffix


        matfile = read_matfile(os.path.join(self.img_dir_path, file_path_split[-2], file_without_suffix + '.mat'))
        max_rf = matfile['max_rf'].astype(np.float32)
        label = matfile['IDH_patient'].astype(np.long)
        mask = np.zeros(max_rf.shape)
        print('mask', mask.shape)



        # train时随机取self.patch_view个图像块，test时取all数据块
        with open(file_path, 'r') as f:
            jsonfile = json.load(f)
            patch_list = jsonfile[k]

            selected_patch_list = patch_list #测试集的所有数据块

            img_file_stack = []

            # 提取图像块并预处理
            for pp in selected_patch_list:

                _, x, y, patch_length, patch_width, _, _, maxv, minv = pp

                # patch_label = self.patch_class[patch_label]

                data = max_rf[x:x+patch_length, y:y+patch_width]

                img_transform = self.my_transforms((data, maxv, minv) )

                img_file_stack.append(img_transform)


            img_file_stack = torch.stack(img_file_stack, dim=0)
            return img_file_stack, np.array(label), file_without_suffix, mask, selected_patch_list

    def __len__(self):
        return len(self.json_files_path)

def read_matfile(matfile_path):
    """读取matfile中的文件, 并返回

    Args:
        matfile_path (str): matfile的绝对路径

    Returns:
        dict: 返回需要返回的内容
    """
    try:
        matfile = sio.loadmat(matfile_path)
        max_rf = matfile['max_rf']
        mask = matfile['roi']
        label_IDH = matfile['IDH_patient'][0][0]
        label_1p19q = matfile['X1p19q_patient'][0][0]
        label_TERT = matfile['TERT_patient'][0][0]
    except:
        matfile = h5py.File(matfile_path, 'r')
        max_rf = np.array(matfile['max_rf']).T
        mask = np.array(matfile['roi']).T
        label_IDH = matfile['IDH_patient'][0][0]
        label_1p19q = matfile['X1p19q_patient'][0][0]
        label_TERT = matfile['TERT_patient'][0][0]
    
    shape_max_rf = max_rf.shape
    max_v = max_rf.max()
    min_v = max_rf.min()
    return {'max_rf': max_rf, 'mask': mask, 'IDH_patient': label_IDH, 'X1p19q_patient': label_1p19q, 'TERT_patient': label_TERT,
             "shape": shape_max_rf, "max_val": max_v, "min_val": min_v}


def get_ROI_patches(patch_list):
    roi_list = []
    for pp in patch_list:
        _, x, y, patch_length, patch_width, patch_label, _, maxv, minv = pp #获取详细信息

        if patch_label != "bg":
            roi_list.append(pp)

    return roi_list