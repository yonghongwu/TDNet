from math import floor
from torchvision import transforms
from utils.RandAug import RandAugment
from torch.utils.data import ConcatDataset
import argparse

import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../'))

from util_dataset import SimpleDataSet, ComplexDataSet
from util_dataset import SimpleDataSet_unlabel, ComplexDataSet_unlabel

def raiseerror(x):
    raise ValueError(x)


transform = {
    "None": None,
    "Weak": transforms.Compose([transforms.RandomResizedCrop(512, scale=(0.95, 1)),
                                transforms.RandomHorizontalFlip(p=0.5)]),
    "PlusWeak": transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)]),
    "Strong": [transforms.Compose([transforms.RandomRotation(20),
                                  transforms.RandomResizedCrop(512, scale=(0.85, 1))]),
               transforms.RandomErasing(p=0.3, ratio=(0.6, 1.66), scale=(0.02, 0.05), value=0)],
    "un_use": RandAugment(n=2, m=2)} # RA(PIL.Image)


def loader(mode='simple', args=None, shuffle=True):
    # ['lushulin' 'wangdaiping' 'zouzhengsheng']
    A, B, C = args.batchsize//3, args.batchsize//3, args.batchsize - args.batchsize//3-args.batchsize//3
    D, E, F = args.batchsize//6, args.batchsize//6, args.batchsize//2-args.batchsize//6-args.batchsize//6
    data_ori = '../../../../storage/wuyonghuang/vessel_seg/all_data_process'
    label_ori = '../../../../storage/wuyonghuang/vessel_seg/all_label_process'
    data_unlabel_ori = '../../../../storage/wuyonghuang/vessel_seg/all_data_unlabel_process'
    data_unlabel_noise_ori = '../../../../storage/wuyonghuang/vessel_seg/all_data_unlabel_process_noise'
    check = [raiseerror('path-{} dont exists!'.format(i)) for i in [data_ori, label_ori, data_unlabel_ori] if
             not os.path.exists(i)]

    s = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, tr_te_param=3)
    tr_pat_name, te_pat_name = s.tr_pat_name, s.te_pat_name

    part_tr_num = floor(len(tr_pat_name) * args.ratio)
    part_tr_name = np.random.choice(tr_pat_name, size=part_tr_num, replace=False)  # replacement=True means it will be selected one time.
    res_tr_name = [i for i in tr_pat_name if i not in part_tr_name] if len(tr_pat_name) != len(part_tr_name) else None

    print("the num of labeled_pat in train is: ", len(part_tr_name))
    print("the num of labeled_pat in test  is: ", len(te_pat_name), "; te_pat_name: ", te_pat_name, '\n')

    s_un = SimpleDataSet_unlabel(data_path_ori=data_unlabel_ori)
    unlabeled_pat_name = s_un.tr_pat_name
    print('the num of unlabeled_pat is: {}.'.format(len(unlabeled_pat_name)))
    # ---------------------------------------------------------------
    #'simple': # Done: simple and complex loader are all be needed. use iter() and next(); about length.
    if mode == 'simple':
        simpleD_labeled = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                        tr_pat_name=part_tr_name, te_pat_name=te_pat_name, typ='train', transform=transform['Weak'])
        simpleD_unlabeled = SimpleDataSet_unlabel(data_path_ori=data_unlabel_ori, is_cuda=True, transform=transform['Weak'])
        if args.add_noise:
            a = SimpleDataSet_unlabel(data_path_ori=data_unlabel_noise_ori, is_cuda=True, transform=transform['Weak'])
            simpleD_unlabeled += a
        res_labeled = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                            tr_pat_name=res_tr_name, te_pat_name=te_pat_name, typ='train',
                            transform=transform['Weak'], res=True) if res_tr_name is not None else None

        if res_labeled is not None:
            simpleD_labeled_loader = torch.utils.data.DataLoader(simpleD_labeled, batch_size=A,
                                                                 num_workers=args.num_workers, pin_memory=False, shuffle=shuffle)
            s_res_loader = torch.utils.data.DataLoader(res_labeled, batch_size=B, num_workers=args.num_workers, pin_memory=False, shuffle=shuffle)
            assert A + B + C == args.batchsize, 'not match'
        else:
            simpleD_labeled_loader = torch.utils.data.DataLoader(simpleD_labeled, batch_size=A,
                                                                 num_workers=args.num_workers, pin_memory=False, shuffle=shuffle)
            s_res_loader = None
            C += B
            assert A + C == args.batchsize, 'not match'

        simpleD_unlabeled_loader = torch.utils.data.DataLoader(simpleD_unlabeled, batch_size=C,
                                                               num_workers=args.num_workers, pin_memory=False, shuffle=shuffle)
        # 'validloader'
        validset = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                 tr_pat_name=part_tr_name, te_pat_name=te_pat_name, typ='test',
                                 transform=transform['Weak'])
        validloader = torch.utils.data.DataLoader(validset, batch_size=args.batchsize, num_workers=args.num_workers, pin_memory=False, shuffle=shuffle)

        Cdataset = SimpleDataSet(data_path_ori=data_ori, labels_path_ori=label_ori, is_cuda=True,
                                 tr_pat_name=res_tr_name[:10], te_pat_name=te_pat_name, typ='train',
                                 transform=transform['Weak'])
        Cloader = torch.utils.data.DataLoader(Cdataset, batch_size=args.batchsize, num_workers=args.num_workers, pin_memory=False, shuffle=shuffle)

        return simpleD_labeled_loader, simpleD_unlabeled_loader, s_res_loader, validloader, Cloader
