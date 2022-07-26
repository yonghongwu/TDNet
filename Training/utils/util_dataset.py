from typing import Tuple, Union

import numpy as np
import pandas as pd
import pydicom
import os

from typing import Union
from numpy import ndarray
from pandas.core.arrays import ExtensionArray
from torch.utils.data import Dataset
import nibabel as nib
import torch
from typing import Union
from math import floor
import hdf5storage as hdf
import albumentations as al
import matplotlib.pyplot as plt


def get_data_and_label_path(data_path='./va_data', label_path='./va_label'):
    """
    The file arch should be :
    --va_data:
        --chengang
            --12
                --dicom 1
                --dicom 2
                --...
            --13
                --dicom 1
                --dicom 2
                --...
            --...
    --va_label:
        --chengang
            --12.nii
            --13.nii
            --...
    :param data_path: default = './va_data'
    :param label_path: default = './va_label'
    :return: data_path_all: the path to 'dicom file'
            label_path_all: the path to 'nii file'
    """
    va_data_path = data_path
    va_label_path = label_path

    va_name = os.listdir(va_data_path)  # listdir example:[chengang,...]
    va_name.sort()

    va_path_all = []
    data_path_all = []
    label_path_all = []
    for i in range(len(va_name)):
        va_path = os.path.join(va_data_path, va_name[i])
        va_path_all.append(va_path)
        va_path_dir = os.listdir(va_path + "/CT_sequence")  # example:[12,...]
        va_path_dir.sort()
        for j in range(len(va_path_dir)):
            data_path = os.path.join(va_path, "CT_sequence", va_path_dir[j])
            data_path_all.append(data_path)

        for k in range(len(va_path_dir)):
            label_path = os.path.join(os.path.join(va_label_path, va_name[i]), va_path_dir[k])
            label_path_all.append(label_path)

    return data_path_all, label_path_all

def get_data(data_pathh):
    data_result = np.zeros((512, 512, 160))
    data_pathh_dir = os.listdir(data_pathh)
    data_pathh_dir.sort()  # linux的文件排序不同， 导致文件不匹配
    for kk in range(160):
        numm = len(data_pathh_dir) - 160 + kk
        dic = pydicom.read_file(os.path.join(data_pathh, data_pathh_dir[numm]))
        data_result[:, :, kk] = dic.pixel_array + dic.RescaleIntercept
    # data_result[data_result > 600] = 600
    data_result[data_result < 0] = 0
    return data_result

class MyDataSet(Dataset):
    """
    :return inputs_: (512, 512, 160)
    :return labels_: (512, 512, 319)
    """

    def __init__(self, data_path_ori, labels_path_ori, transform=None, dev='linux'):
        super(MyDataSet, self).__init__()
        self.data_ori = data_path_ori
        self.labels_ori = labels_path_ori
        self.transform = transform
        self.dev = dev
        self.data_path, self.labels_path = get_data_and_label_path(data_path=self.data_ori, label_path=self.labels_ori)

    def __getitem__(self, idx):
        inputs_ = get_data(self.data_path[idx])  # size=(512,512,160) exam:'D:/wuyonghuang/git/F/te_data/chengang/12'
        if self.dev == "linux":
            labels_ = nib.load(self.labels_path[idx] + '.nii').get_fdata().astype(np.float32)  # size=512,512,319
            labels_ = np.flip(labels_, axis=0)
            labels_ = np.rot90(labels_, -1, axes=(0, 1))

            affine_ = nib.load(self.labels_path[idx] + '.nii').affine
        elif self.dev == "windows" or "win":
            labels_ = nib.load(self.labels_path[idx] + '.nii').get_fdata().astype(np.float32)  # size=512,512,319
            labels_ = np.flip(labels_, axis=0)
            labels_ = np.rot90(labels_, -1, axes=(0, 1))

            affine_ = nib.load(self.labels_path[idx] + '.nii').affine
        else:
            raise ValueError("self.dev must be 'windows/win' or 'linux'.")

        return inputs_, labels_, affine_

    def __len__(self):
        return len(self.data_path)

class MyDataSet2(Dataset):
    """
    用于complexNet的dataloader
    """
    def __init__(self, data_path_ori='../../../storage/wuyonghuang/vessel_seg/train_data',
                 labels_path_ori='../../../storage/wuyonghuang/vessel_seg/train_label', transform=None,
                 mybatchsize=32, is_repeat: int = 2):
        super(MyDataSet2, self).__init__()
        self.myBatchSize = mybatchsize
        self.data_path_ori = data_path_ori
        self.labels_path_ori = labels_path_ori
        self.transform = transform
        self.data_folder_lst = os.listdir(data_path_ori)
        self.data_folder_lst.sort()
        self.label_folder_lst = os.listdir(labels_path_ori)
        self.label_folder_lst.sort()
        self.is_repeat = is_repeat

    def __getitem__(self, idx):
        idx = idx // self.is_repeat
        deci_folder_name = np.random.choice(self.data_folder_lst)
        data_folder_pth = os.path.join(self.data_path_ori, deci_folder_name)
        label_folder_pth = os.path.join(self.labels_path_ori, deci_folder_name)
        data_lst = os.listdir(data_folder_pth)
        label_lst = os.listdir(label_folder_pth)
        data_lst.sort()
        label_lst.sort()

        concat_data = []
        concat_label = []
        for i in range(self.myBatchSize):
            data_pth = os.path.join(data_folder_pth, data_lst[idx+i])
            label_pth = os.path.join(label_folder_pth, label_lst[idx+i])

            data = torch.unsqueeze(torch.from_numpy(np.load(data_pth)), 0)
            label = torch.unsqueeze(torch.from_numpy(np.load(label_pth)), 0)
            concat_data.append(data)
            concat_label.append(label)
        final_data = torch.cat(concat_data, dim=0)
        final_label = torch.cat(concat_label, dim=0)
        return final_data, final_label

    def __len__(self):
        return (160-self.myBatchSize) // self.myBatchSize * self.is_repeat
    # train的时候原本的epoch应该再乘以总的病人的时刻数

class SimpleDataSet(Dataset):
    """
    用于simpleNet的dataloader
    """
    def __init__(self, data_path_ori='../../../storage/wuyonghuang/vessel_seg/all_data_process',
                 labels_path_ori='../../../storage/wuyonghuang/vessel_seg/all_label_process',
                 transform=None, is_cuda=True, typ='train',
                 tr_te_param: Union[int, float]=3, tr_pat_name=None, te_pat_name=None, res=False):
        super(SimpleDataSet, self).__init__()
        self.is_cuda = is_cuda
        self.data_path_ori = data_path_ori
        self.labels_path_ori = labels_path_ori
        self.transform = transform
        self.res = res
        self.data_lst = os.listdir(data_path_ori)
        self.data_lst.sort()
        self.label_lst = os.listdir(labels_path_ori)
        self.label_lst.sort()

        self.num, self.name = self.patient_num()
        if tr_pat_name is None and te_pat_name is None: # init for splitting trainset and testset.
            if isinstance(tr_te_param, int):
                self.te_pat_name = np.random.choice(self.name, size=tr_te_param, replace=False)
                self.tr_pat_name = [i for i in self.name if i not in self.te_pat_name]
            elif isinstance(tr_te_param, float):
                choice_size = floor(self.num * tr_te_param)
                self.te_pat_name = np.random.choice(self.name, size=choice_size, replace=False)
                self.tr_pat_name = [i for i in self.name if i not in self.te_pat_name]

        elif tr_pat_name is not None and te_pat_name is not None:   # get dataset
            self.tr_pat_name = tr_pat_name
            self.te_pat_name = te_pat_name
            if typ == 'train':
                self.data_lst = self.filter(self.data_lst, self.tr_pat_name)
                self.label_lst = self.filter(self.label_lst, self.tr_pat_name)
            elif typ == 'test':
                self.data_lst = self.filter(self.data_lst, self.te_pat_name)
                self.label_lst = self.filter(self.label_lst, self.te_pat_name)

    def __getitem__(self, idx):
        data_pth = os.path.join(self.data_path_ori, self.data_lst[idx])  # image_path='./patient001_11_160.npy'
        label_pth = os.path.join(self.labels_path_ori, self.label_lst[idx])

        data = np.load(data_pth)
        data[data>1000] = 1000
        data[data<10] = 0

        data = torch.unsqueeze(torch.from_numpy(data), dim=0)
        label = torch.unsqueeze(torch.from_numpy(np.load(label_pth)), dim=0)
        if self.transform:  # 是否使用rand augmentation，使用方法见./utils/RandAug.py
            al = torch.concat([data, label], dim=0)     # if you use ColorJitter , it will scale to 0-1.
            trans_al = self.transform(al).chunk(2, dim=0)
            data, label  = trans_al
            # [[plt.imshow(i.cpu().numpy()[0, :, :]), plt.show()] for i in trans_al]
        if self.is_cuda:
            data = data.float().cuda()
            label = label.long().cuda()
        if self.res is False:
            mark = torch.Tensor([0])    # 意味着是source domain上的
        else:
            mark = torch.Tensor([1])    # 意味着是target domain上的
        return data, label, mark

    def __len__(self):
        return len(self.data_lst)

    def patient_num(self) -> Tuple[int, Union[ExtensionArray, ndarray]]:
        a = [os.path.splitext(i)[0] for i in self.data_lst]
        temp = pd.Series(a)
        temp1 = temp.apply(lambda x: x.split('_')[1])
        return len(temp1.unique()), temp1.unique()

    def filter(self, a, b) -> list:
        return [i for i in a if os.path.splitext(i)[0].split('_')[1] in b]

class ComplexDataSet(Dataset):
    """
    用于complexNet的dataloader
    """
    def __init__(self, data_path_ori='../../../storage/wuyonghuang/vessel_seg/all_data_process',
                 labels_path_ori='../../../storage/wuyonghuang/vessel_seg/all_label_process', transform=None,
                 mybatchsize=32, is_cuda=True, typ='train', # type='train' or 'test'
                 tr_te_param: Union[int, float]=3, tr_pat_name=None, te_pat_name=None, res=False):
        super(ComplexDataSet, self).__init__()
        self.myBatchSize = mybatchsize
        self.transform = transform
        self.data_path_ori = data_path_ori
        self.labels_path_ori = labels_path_ori
        self.res = res

        self.data_lst = os.listdir(data_path_ori)
        self.data_lst.sort()
        self.data_lst = pd.Series(self.data_lst).apply(lambda x: os.path.splitext(x)[0]).values

        self.label_lst = os.listdir(labels_path_ori)
        self.label_lst.sort()
        self.label_lst = pd.Series(self.label_lst).apply(lambda x: os.path.splitext(x)[0]).values
        self.is_cuda = is_cuda
        self.typ = typ

        self.num, self.name = self.patient_num()
        if tr_pat_name is None and te_pat_name is None:  # init for splitting trainset and testset.
            if isinstance(tr_te_param, int):
                self.te_pat_name = np.random.choice(self.name, size=tr_te_param, replace=False)
                self.tr_pat_name = [i for i in self.name if i not in self.te_pat_name]
            elif isinstance(tr_te_param, float):
                choice_size = floor(self.num * tr_te_param)
                self.te_pat_name = np.random.choice(self.name, size=choice_size, replace=False)
                self.tr_pat_name = [i for i in self.name if i not in self.te_pat_name]

        elif tr_pat_name is not None and te_pat_name is not None:  # get dataset
            self.tr_pat_name = tr_pat_name
            self.te_pat_name = te_pat_name
            if self.typ == 'train':
                self.data_lst = self.filter(self.data_lst, self.tr_pat_name)
                self.label_lst = self.filter(self.label_lst, self.tr_pat_name)
            elif self.typ == 'test':
                self.data_lst = self.filter(self.data_lst, self.te_pat_name)
                self.label_lst = self.filter(self.label_lst, self.te_pat_name)

    def __getitem__(self, idx):
        d = int(self.data_lst[idx].split('_')[-1]) > 160-self.myBatchSize
        # ceshi_data = []
        # ceshi_label = []
        if d:
            da_string = '_'.join(self.data_lst[idx].split('_')[:-1])
            la_string = '_'.join(self.label_lst[idx].split('_')[:-1])
            con_data = []
            con_label = []
            for i in range(self.myBatchSize):
                next_da_string = da_string + '_' + '{:03d}'.format(160-self.myBatchSize+i)
                next_la_string = la_string + '_' + '{:03d}'.format(160-self.myBatchSize+i)
                da_pth = os.path.join(self.data_path_ori, next_da_string) + ".npy"
                la_pth = os.path.join(self.labels_path_ori, next_la_string) + ".npy"

                data = np.load(da_pth)
                data[data > 1000] = 1000
                data[data < 10] = 0

                da = torch.unsqueeze(torch.from_numpy(data), dim=0)
                la = torch.unsqueeze(torch.from_numpy(np.load(la_pth)), dim=0)
                con_data.append(da)
                con_label.append(la)
                # ceshi_data.append(da_pth)
                # ceshi_label.append(la_pth)
            data = torch.cat(con_data, dim=0)
            label = torch.cat(con_label, dim=0)
        else:
            con_data = []
            con_label = []
            for i in range(self.myBatchSize):
                da_pth = os.path.join(self.data_path_ori, self.data_lst[idx+i]) + ".npy"
                la_pth = os.path.join(self.labels_path_ori, self.label_lst[idx+i]) + ".npy"

                data = np.load(da_pth)
                data[data > 1000] = 1000
                data[data < 0] = 0

                da = torch.unsqueeze(torch.from_numpy(data), dim=0)
                la = torch.unsqueeze(torch.from_numpy(np.load(la_pth)), dim=0)
                con_data.append(da)
                con_label.append(la)
                # ceshi_data.append(da_pth)
                # ceshi_label.append(la_pth)
            data = torch.cat(con_data, dim=0)
            label = torch.cat(con_label, dim=0)
        if self.is_cuda:
            data = data.float().cuda()
            label = label.long().cuda()
        if self.transform is not None:
            al = torch.concat([data, label], dim=0)
            trans_al = self.transform(al).chunk(2, dim=0)
            data, label = trans_al
        if self.res is False:
            mark = torch.Tensor([0])    # 意味着在source domain 上
        else:
            mark = torch.Tensor([1])    # 意味着在target domain上
        return data, label, mark  # , ceshi_data, ceshi_label

    def __len__(self):
        if self.typ == 'train':
            return 160 * len(self.tr_pat_name)
        elif self.typ == 'test':
            return 160 * len(self.te_pat_name)

    def patient_num(self) -> Tuple[int, Union[ExtensionArray, ndarray]]:
        a = [os.path.splitext(i)[0] for i in self.data_lst]
        temp = pd.Series(a)
        temp1 = temp.apply(lambda x: x.split('_')[1])
        return len(temp1.unique()), temp1.unique()

    def filter(self, a, b) -> list:
        return [i for i in a if os.path.splitext(i)[0].split('_')[1] in b]

class SimpleDataSet_unlabel(Dataset):
    def __init__(self, data_path_ori='../../../storage/wuyonghuang/vessel_seg/all_data_unlabel_process',
                 transform=None, is_cuda=True, tr_pat_name=None):
        super(SimpleDataSet_unlabel, self).__init__()
        self.is_cuda = is_cuda
        self.data_path_ori = data_path_ori
        self.transform = transform
        self.data_lst = os.listdir(data_path_ori)
        self.data_lst.sort()

        self.num, self.name = self.patient_num()
        if tr_pat_name is None: # init for splitting trainset and testset.
            self.tr_pat_name = [i for i in self.name]

    def __getitem__(self, idx):
        data_pth = os.path.join(self.data_path_ori, self.data_lst[idx])  # image_path='./patient001_11_160.npy'

        data = np.load(data_pth)
        data[data > 1000] = 1000
        data[data < 10] = 0

        data = torch.unsqueeze(torch.from_numpy(data), dim=0)
        if self.transform:  # 是否使用rand augmentation，使用方法见./utils/RandAug.py
            data = self.transform(data)
        if self.is_cuda:
            data = data.float().cuda()
            label = torch.zeros_like(data).long().cuda()
        else:
            label = torch.zeros_like(data).long()
        return data, label, torch.Tensor([1])  # means that is unlabeled data.

    def __len__(self):
        return len(self.data_lst)

    def patient_num(self) -> Tuple[int, Union[ExtensionArray, ndarray]]:
        a = [os.path.splitext(i)[0] for i in self.data_lst]
        temp = pd.Series(a)
        temp1 = temp.apply(lambda x: x.split('_')[1])
        return len(temp1.unique()), temp1.unique()

    def filter(self, a, b) -> list:
        return [i for i in a if os.path.splitext(i)[0].split('_')[1] in b]

class ComplexDataSet_unlabel(Dataset):
    def __init__(self, data_path_ori='../../../storage/wuyonghuang/vessel_seg/all_data_process',
                 transform=None, mybatchsize=32, is_cuda=True, tr_pat_name=None, noise=True):
        super(ComplexDataSet_unlabel, self).__init__()
        self.myBatchSize = mybatchsize
        self.transform = transform
        self.data_path_ori = data_path_ori

        self.data_lst = os.listdir(data_path_ori)
        self.data_lst.sort()
        self.data_lst = pd.Series(self.data_lst).apply(lambda x: os.path.splitext(x)[0]).values

        self.is_cuda = is_cuda

        self.num, self.name = self.patient_num()
        if tr_pat_name is None:  # init for splitting trainset and testset.
            self.tr_pat_name = [i for i in self.name]

        if noise == True:
            self.Q = 40
        elif noise == False:
            self.Q = 110

    def __getitem__(self, idx):
        d = int(self.data_lst[idx].split('_')[-1]) > self.Q-self.myBatchSize
        # ceshi_data = []
        if d:
            da_string = '_'.join(self.data_lst[idx].split('_')[:-1])
            con_data = []
            for i in range(self.myBatchSize):
                next_da_string = da_string + '_' + '{:03d}'.format(self.Q-self.myBatchSize+i)
                da_pth = os.path.join(self.data_path_ori, next_da_string) + ".npy"

                data = np.load(da_pth)
                data[data > 1000] = 1000
                data[data < 10] = 0

                da = torch.unsqueeze(torch.from_numpy(data), dim=0)
                con_data.append(da)
                # ceshi_data.append(da_pth)
            data = torch.cat(con_data, dim=0)
        else:
            con_data = []
            for i in range(self.myBatchSize):
                da_pth = os.path.join(self.data_path_ori, self.data_lst[idx+i]) + ".npy"

                data = np.load(da_pth)
                data[data > 1000] = 1000
                data[data < 0] = 0

                da = torch.unsqueeze(torch.from_numpy(data), dim=0)
                con_data.append(da)
                # ceshi_data.append(da_pth)
            data = torch.cat(con_data, dim=0)
        if self.is_cuda:
            data = data.float().cuda()
            label = torch.zeros_like(data).long().cuda()
        else:
            label = torch.zeros_like(data).long()

        if self.transform:
            data = self.transform(data)
        return data, label, torch.Tensor([1])  # , ceshi_data, ceshi_label

    def __len__(self):
        return self.Q * len(self.tr_pat_name)

    def patient_num(self) -> Tuple[int, Union[ExtensionArray, ndarray]]:
        a = [os.path.splitext(i)[0] for i in self.data_lst]
        temp = pd.Series(a)
        temp1 = temp.apply(lambda x: x.split('_')[1])
        return len(temp1.unique()), temp1.unique()

    def filter(self, a, b) -> list:
        return [i for i in a if os.path.splitext(i)[0].split('_')[1] in b]

class VisDataSet(Dataset):
    """
    用于预测lushulin的血管，论文展示.
    """
    def __init__(self, data_paths:list, label_paths:list, transform=None, is_cuda=True,):
        super(VisDataSet, self).__init__()
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.transform = transform
        self.is_cuda = is_cuda

    def __getitem__(self, idx):
        data=[]
        label=[]

        for i in range(len(self.data_paths)):
            data_i = np.load(self.data_paths[i])
            data_i[data_i>1000] = 1000
            data_i[data_i<10] = 0

            data_i = torch.unsqueeze(torch.from_numpy(data_i), dim=0)
            label_i = torch.unsqueeze(torch.from_numpy(np.load(self.label_paths[i])), dim=0)

            data.append(data_i)
            label.append(label_i)

        data = torch.concat(data, dim=0)
        label = torch.concat(label, dim=0)

        if self.transform:  # 是否使用rand augmentation，使用方法见./utils/RandAug.py
            al = torch.concat([data, label], dim=0)     # if you use ColorJitter , it will scale to 0-1.
            trans_al = self.transform(al).chunk(2, dim=0)
            data, label  = trans_al
            # import matplotlib.pyplot as plt
            # [[plt.imshow(i.cpu().numpy()[0, :, :]), plt.show()] for i in trans_al]
        if self.is_cuda:
            data = data.float().cuda()
            label = label.long().cuda()
        return data, label

    def __len__(self):
        # assert len(self.data_paths) == len(self.label_paths), 'check.'
        # return len(self.data_paths)
        return 1

class air_dataset(Dataset):
    def __init__(self, data_root, label_root, num_slice, transforms, typ='train', ratio=0.05, cuda=False, res=False, Exres_lst=None, clip=True):
        '''
        >>> data_root  = '../Train/imagesTr' # Train or Valid
        >>> label_root = '../Train/labelsTr'    # Train or Valid
        >>> num_slice = args.batchsize/2
        '''
        self.data_root  = data_root
        self.label_root = label_root
        self.num_slice  = num_slice
        self.transforms = transforms
        self.typ        = typ
        self.ratio = ratio
        self.cuda = cuda
        self.res = res  # construct res_loader: same distribution like train_set, but dont give label
        self.clip = clip

        assert self.typ in ['train', 'test', 'semi']
        if self.typ in ['train', 'test']:
            self.data  = os.listdir(self.data_root)
            self.label = os.listdir(self.label_root)
            self.data.sort()
            self.label.sort()

            # process: train or test
            self.data_test = self.data[::6]
            self.label_test = self.label[::6]
            self.data_train_all = [i for i in self.data if i not in self.data_test]
            self.label_train_all = [i for i in self.data if i not in self.label_test]

            # ratio
            if self.res is False:
                assert Exres_lst is None
                self.data_train_num = floor(len(self.data_train_all) * self.ratio)
                np.random.seed(0)
                self.data_train = np.random.choice(self.data_train_all, size=self.data_train_num, replace=False)
                np.random.seed(0)
                self.label_train = np.random.choice(self.label_train_all, size=self.data_train_num, replace=False)
            else:
                assert self.res is True and Exres_lst is not None
                Exres_lst = [i.split('/')[-1] for i in Exres_lst]
                self.data_train = [i for i in self.data_train_all if i not in Exres_lst]
                self.label_train = [i for i in self.label_train_all if i not in Exres_lst]

            self.data_train  = [os.path.join(self.data_root, i) for i in self.data_train]
            self.label_train = [os.path.join(self.label_root, i) for i in self.label_train]
            self.data_test = [os.path.join(self.data_root, i) for i in self.data_test]
            self.label_test = [os.path.join(self.label_root, i) for i in self.label_test]

            np.random.seed(10)
            np.random.shuffle(self.data_train)
            np.random.seed(10)
            np.random.shuffle(self.label_train)
            np.random.seed(10)
            np.random.shuffle(self.data_test)
            np.random.seed(10)
            np.random.shuffle(self.label_test)

        else:
            # self.typ = 'semi'
            self.data  = os.listdir(self.data_root)
            self.data.sort()
            self.data_train_all = self.data
            self.data_train = self.data_train_all
            self.data_train = [os.path.join(self.data_root, i) for i in self.data_train]

    def __getitem__(self, item):
        if self.typ == 'train':
            return self._getitem_4train(item)
        elif self.typ == 'test':
            return self._getitem_4test(item)
        else:
            assert self.typ == 'semi'
            return self._getitem_4semi(item)

    def _getitem_4train(self, item):
        data = nib.load(self.data_train[item]).get_fdata()
        label = nib.load(self.label_train[item]).get_fdata()

        data[data>0] = 0
        data[data<-1000] = -1000
        data += 1000
        data /= 1000.
        data  = torch.from_numpy(data)[None, ...]
        label = torch.from_numpy(label)[None, ...]

        if self.cuda:
            data, label = data.cuda(), label.cuda()
        if self.clip:
            _, H, W, D = data.shape
            data = data[:, :, :, int(0.12*D):]
            label = label[:, :, :, int(0.12*D):]
            _, H, W, D = data.shape
            data  = data[:, :, :, int(-(D // self.num_slice * self.num_slice)):]
            label = label[:, :, :, int(-(D // self.num_slice * self.num_slice)):]
        data_lst = []
        label_lst = []
        _, H, W, D = data.shape
        assert int(D//(self.num_slice//2)) != 0, 'Error: {}, {}'.format(D, self.data_train[item])
        if self.transforms is not None:
            for i in range(int(D//(self.num_slice//2))):
                augmented = self.transforms(image=data[0, :, :, int(i*(self.num_slice//2)):int((i+1)*(self.num_slice//2))].numpy(),
                                            mask=label[0, :, :, int(i*(self.num_slice//2)):int((i+1)*(self.num_slice//2))].numpy())
                data_lst.append(torch.from_numpy(augmented['image']))
                label_lst.append(torch.from_numpy(augmented['mask']))
            data = torch.concat(data_lst, dim=-1).unsqueeze(0)
            label = torch.concat(label_lst, dim=-1).unsqueeze(0)
        return data, label

    def _getitem_4test(self, item):
        data = nib.load(self.data_test[item]).get_fdata()
        label = nib.load(self.label_test[item]).get_fdata()
        data[data > 0] = 0
        data[data < -1000] = -1000
        data += 1000
        data /= 1000.
        if self.transforms is not None:
            augmented = self.transforms(image=data, mask=label)
            data, label = augmented['image'], augmented['mask']
        data = torch.from_numpy(data)[None, ...]
        label = torch.from_numpy(label)[None, ...]
        if self.cuda:
            data, label = data.cuda(), label.cuda()
        if self.clip:
            _, H, W, D = data.shape
            data = data[:, :, :, int(0.12 * D):]
            label = label[:, :, :, int(0.12 * D):]
            _, H, W, D = data.shape
            data   = data[:, :, :, int(-(D // self.num_slice * self.num_slice)):]
            label  = label[:, :, :, int(-(D // self.num_slice * self.num_slice)):]
        return data, label

    def _getitem_4semi(self, item):
        data = nib.load(self.data_train[item]).get_fdata()
        data[data > 0] = 0
        data[data < -1000] = -1000
        data += 1000
        data /= 1000.
        if self.transforms is not None:
            augmented = self.transforms(image=data)
            data = augmented['image']
        data = torch.from_numpy(data)[None, ...]
        if self.clip:
            _, H, W, D = data.shape
            data = data[:, :, :, int(-(D // self.num_slice * self.num_slice)):]
        data_lst = []
        _, H, W, D = data.shape
        if self.transforms is not None:
            for i in range(int(D // self.num_slice)):
                augmented = self.transforms(
                    image=data[0, :, :, int(i * self.num_slice):int((i + 1) * self.num_slice)].numpy())
                data_lst.append(torch.from_numpy(augmented['image']))
            data = torch.concat(data_lst, dim=-1).unsqueeze(0)
        label = torch.zeros_like(data)
        if self.cuda:
            data, label = data.cuda(), label.cuda()
        return data, label

    def __len__(self):
        if self.typ == 'train':
            assert len(self.data) == len(self.label), 'Error!'
            return len(self.data_train)
        elif self.typ == 'test':
            assert len(self.data) == len(self.label), 'Error!'
            return len(self.data_test)
        else:
            assert self.typ == 'semi'
            return len(self.data_train)


def collate(data):
    img = torch.concat([item[0] for item in data], dim=-1).permute(0, 3, 1, 2)
    label = torch.concat([item[1] for item in data], dim=-1).permute(0, 3, 1, 2)
    B, D, H, W = img.shape
    assert label.shape == (B, D, H, W)
    return img,label


def collate_semi(data):
    img = torch.concat([item for item in data], dim=-1).permute(0, 3, 1, 2)
    B, D, H, W = img.shape
    img = img.view(B, -1, 16, H, W)
    return img


def collate_2(data):
    img = torch.concat([item[0] for item in data], dim=0)
    lab = torch.concat([item[1] for item in data], dim=0)
    mar = torch.concat([item[2] for item in data], dim=0)
    return img, lab, mar


class air_dataset_2(Dataset):
    r"""
    train_path: 存放具有标注、充当半监督有标注的那一部分数据
    test_path: 存放用于测试的那一部分数据
    unlabel_path: 存放与test_path相同的一批数据，充当半监督中的事实无标注数据
    filenames_res: 具有标注、充当半监督中的逻辑无标注数据
    """
    def __init__(self, args, train_path=None, test_path=None, unlabel_path=None, semi=True, transform: Union[al.Compose, None] = None, para=16):
        self.args = args
        self.ori_path = '../../train_data' if train_path is None else train_path
        self.test_path = '../../test_data' if test_path is None else test_path
        self.unlabel_path = '../../unlabel_process' if unlabel_path is None else unlabel_path
        self.filenames = os.listdir(self.ori_path)
        self.filenames.sort()
        self.filenames_unlabel = os.listdir(self.unlabel_path)
        self.filenames_unlabel.sort()
        self.filenames_test = os.listdir(self.test_path)
        self.filenames_test.sort()

        np.random.seed(args.seed)
        np.random.shuffle(self.filenames)
        self.filenames_labeled = self.filenames[:int(len(self.filenames) * args.ratio)]
        self.filenames_res = self.filenames[int(len(self.filenames) * args.ratio):]

        np.random.seed(args.seed)
        np.random.shuffle(self.filenames_unlabel)

        self.semi = semi
        self.transform = transform
        self.para = para

    def __len__(self):
        return len(self.filenames_labeled)

    def __getitem__(self, item):
        if self.semi:
            if self.args.ratio != 1:
                data1, label1 = self.mat2data(self.filenames_labeled[item])
                mark1 = torch.Tensor([0]).reshape(1, 1).repeat(data1.shape[0], 1)

                data2, label2 = self.mat2data(self.filenames_res[item%(len(self.filenames_res))])
                mark2 = torch.Tensor([1]).reshape(1, 1).repeat(data2.shape[0], 1)

                data3, label3 = self.mat2data_2(self.filenames_unlabel[item%len(self.filenames_unlabel)])
                mark3 = torch.Tensor([1]).reshape(1, 1).repeat(data3.shape[0], 1)

                data = torch.concat([data1, data2, data3], dim=0)
                label = torch.concat([label1, label2, label3], dim=0)
                mark = torch.concat([mark1, mark2, mark3], dim=0)

                return data, label, mark
            else:
                data1, label1 = self.mat2data(self.filenames_labeled[item])
                mark1 = torch.Tensor([0]).reshape(1, 1).repeat(data1.shape[0], 1)

                data2, label2 = self.mat2data_2(self.filenames_unlabel[item % len(self.filenames_unlabel) // 2])
                mark2 = torch.Tensor([1]).reshape(1, 1).repeat(data2.shape[0], 1)

                data3, label3 = self.mat2data_2(self.filenames_unlabel[item % len(self.filenames_unlabel)])
                mark3 = torch.Tensor([1]).reshape(1, 1).repeat(data3.shape[0], 1)

                data = torch.concat([data1, data2, data3], dim=0)
                label = torch.concat([label1, label2, label3], dim=0)
                mark = torch.concat([mark1, mark2, mark3], dim=0)

                return data, label, mark
        else:
            assert self.semi is False
            data, label = self.mat2data(self.filenames_labeled[item])
            mark = torch.Tensor([0]).reshape(1, 1).repeat(data.shape[0], 1)
            return data, label, mark

    def mat2data(self, p):
        pth = self.ori_path
        in_data = hdf.loadmat(os.path.join(pth, p))
        ima = in_data['data_ima_32']    # (320, 200, 32)
        seg = in_data['data_lab_32']    # (320, 200, 32)

        if self.transform:
            augmented = self.transform(image=ima, mask=seg)
            ima, seg = augmented['image'], augmented['mask']

        fea = np.concatenate((ima, seg), axis=2)
        fea = fea.transpose((2, 0, 1))  # 64, 320, 200

        pats_squ = torch.from_numpy(fea).float().unsqueeze(1)

        inputs = pats_squ[8:24, :, :, 4:196] if self.para == 16 else pats_squ[:32, :, :, 4:196]
        labels = pats_squ[40:56, :, :, 4:196].long() if self.para == 16 else pats_squ[32:64, :, :, 4:196].long()

        return inputs, labels

    def mat2data_2(self, p):
        pth = self.unlabel_path
        in_data = hdf.loadmat(os.path.join(pth, p))
        ima = in_data['data_ima_32']    # (16, 1, 320, 192)

        seg = np.zeros_like(ima)

        inputs = torch.from_numpy(ima).float()
        labels = torch.from_numpy(seg).long()

        return inputs, labels


# example
if __name__ == "__main__1":
    # loader_type = 'simple'
    loader_type = 'complex'
    if loader_type == 'simple':
        data_ori = '../../../storage/wuyonghuang/vessel_seg/all_data_process'
        label_ori = '../../../storage/wuyonghuang/vessel_seg/all_label_process'
        # dataset = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori)
        # trainloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
        s = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, tr_te_param=3)
        tr_pat_name, te_pat_name = s.tr_pat_name, s.te_pat_name

        train_dataset = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                      tr_pat_name=tr_pat_name, te_pat_name=te_pat_name, typ='train')
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)

        valid_dataset = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                      tr_pat_name=tr_pat_name, te_pat_name=te_pat_name, typ='test')
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)

        for batch_id, (data, label) in enumerate(validloader):
            print(data.shape)
            print(label.shape)
    elif loader_type == 'complex':
        data_ori = '../../../storage/wuyonghuang/vessel_seg/all_data_process'
        label_ori = '../../../storage/wuyonghuang/vessel_seg/all_label_process'
        s = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, tr_te_param=3)
        tr_pat_name, te_pat_name = s.tr_pat_name, s.te_pat_name

        train_dataset = ComplexDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                       mybatchsize=8,
                                       tr_pat_name=tr_pat_name, te_pat_name=te_pat_name, typ='train')
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)
        valid_dataset = ComplexDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                       mybatchsize=8,
                                       tr_pat_name=tr_pat_name, te_pat_name=te_pat_name, typ='test')
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)

        for batch_id, (data, label, ceshi_data, ceshi_label) in enumerate(validloader):
            print(batch_id)
            print(data.shape)
            print(label.shape)
    print('Finished!')


if __name__ == "__main__2":
    loader_type = 'simple'
    if loader_type == 'simple':
        data_ori = '../../../storage/wuyonghuang/vessel_seg/all_data_unlabel_process'
        s_un = SimpleDataSet_unlabel(data_path_ori=data_ori)
        unlabeled_pat_name = s_un.tr_pat_name

        train_dataset = SimpleDataSet_unlabel(data_path_ori=data_ori, is_cuda=True)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)

        valid_dataset = SimpleDataSet_unlabel(data_path_ori=data_ori, is_cuda=True)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)

        for batch_id, (data, label, mark) in enumerate(validloader):
            print(data.shape, label.shape)
            print(mark)
            break
    elif loader_type == 'complex':
        data_ori = '../../../storage/wuyonghuang/vessel_seg/all_data_unlabel_process'
        s = SimpleDataSet_unlabel(data_path_ori=data_ori)
        tr_pat_name, te_pat_name = s.tr_pat_name, s.te_pat_name

        train_dataset = ComplexDataSet_unlabel(data_path_ori=data_ori, is_cuda=True, mybatchsize=8)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)
        valid_dataset = ComplexDataSet_unlabel(data_path_ori=data_ori, is_cuda=True, mybatchsize=8)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0,
                                                  pin_memory=False)

        for batch_id, (data, label, mark) in enumerate(validloader):
            print(data.shape, label.shape)
            print(mark)
            break
    print('Finished!')


if __name__ == "__main__3":
    from utils.util_loader import loader
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batchsize = 32
    args.add_noise = False
    args.ratio = 0.5
    args.seq = 10
    args.num_workers = 0
    labeled_loader, unlabeled_loader, res_loader, validloader = loader(mode='complex', args=args)
    labeled_loader = iter(labeled_loader)
    A, B, C = next(labeled_loader)
    import matplotlib.pyplot as plt

    var = [[plt.imshow(A[0, i, :, :].cpu().detach().numpy()), plt.show()] for i in range(8)]
    var2 = [[plt.imshow(B[0, i, :, :].cpu().detach().numpy()), plt.show()] for i in range(8)]


if __name__ == "__main__4":
    from collections import OrderedDict
    args = OrderedDict()
    args['batchsize'] = 48
    args['ratio'] = 0.05
    args['num_workers'] = 0
    # set
    train_set = air_dataset(data_root='../../Train/imagesTr',
                            label_root='../../Train/labelsTr',
                            num_slice=args['batchsize'] / 2,
                            transforms=None,
                            typ='train', ratio=args['ratio'])
    lst = train_set.data_train
    train_set_res = air_dataset(data_root='../../Train/imagesTr',
                            label_root='../../Train/labelsTr',
                            num_slice=args['batchsize'] / 2,
                            transforms=None,
                            typ='train', ratio=args['ratio'], res=True, Exres_lst=lst)
    valid_set = air_dataset(data_root='../../Train/imagesTr',
                            label_root='../../Train/labelsTr',
                            num_slice=args['batchsize'] / 2,
                            transforms=None,
                            typ='test', ratio=args['ratio'])
    # loader
    labeled_loader = torch.utils.data.DataLoader(train_set,
                                                 batch_size=2,  # only means num of patients
                                                 num_workers=args['num_workers'], pin_memory=False, shuffle=True,
                                                 collate_fn=collate,
                                                 )
    res_loader = torch.utils.data.DataLoader(train_set_res,
                                             batch_size=2,
                                             num_workers=args['num_workers'], pin_memory=False, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set,
                                              batch_size=2,
                                              num_workers=args['num_workers'], pin_memory=False, shuffle=False
                                              )

    labeled_loader_ = iter(labeled_loader)
    data, label = next(labeled_loader_)
    B, n, num_slice, H, W = data.shape
    assert data.shape == label.shape
    data = data.reshape(B*n*num_slice, H, W).unsqueeze(1)
    label = label.reshape(B*n*num_slice, H, W).unsqueeze(1)
    print(data.shape, label.shape)

    for i in range(B*n):
        pass


if __name__ == "__main__":
    class parser(object):
        def __init__(self):
            self.ratio = 1.0
            self.seed = 1
    args = parser()
    dataset = air_dataset_2(args=args)
    L = torch.utils.data.DataLoader(dataset, batch_size=torch.cuda.device_count(), shuffle=True, num_workers=0, pin_memory=False, collate_fn=collate_2)
    for i, (data, label, mark) in enumerate(L):
        # data.shape = 16*3*torch.cuda.device_count(), 1, H, W
        print(data.shape, label.shape)  # data could be used: data.view(-1, 16, H, W)
        break
