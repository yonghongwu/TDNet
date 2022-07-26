import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import hdf5storage as hdf
import os
import segmentation_models_pytorch as smp
from torch.nn.parallel.scatter_gather import gather
from collections import OrderedDict
import math
import scipy.io as sio
from tqdm import tqdm
import logging
import albumentations as al
import oyaml as yaml
import shutil

from util import calculate_dice
from ptsemseg.models import get_model
from ptsemseg.loss.losses import DiceLoss
from utils.util import calculate_dice
from utils.CCT_tool import losses
from utils.util_dataset import air_dataset_2, collate_2
from utils.sendEmail import let_me_know


# ------------------模型，优化方法------------------------------
torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 4"


class parser(object):
    def __init__(self):
        pass


def load_from_smp():
    args = parser()
    args.model = 'Unet'
    args.encoder = 'resnet50'
    args.pretrained_weight = 'imagenet'
    args.in_chans = 1
    args.out_chans = 2
    args.activation = 'sigmoid'
    args.lr = 0.0001
    net = smp.Unet(
        encoder_name=args.encoder,
        encoder_depth=5,
        encoder_weights=args.pretrained_weight,
        decoder_use_batchnorm=True,
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type=None,
        in_channels=args.in_chans,
        classes=args.out_chans,
        activation=None,
        aux_params=None,
    )
    return net

args = parser()
args.seed = 0
args.epochs = 50
args.earlystop = 5
args.lr = 5e-4  # Note.
args.ratio = 1.0
args.batchsize = torch.cuda.device_count()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

res_name = '../../result/TDNet_' + str(args.ratio)
if not os.path.exists(res_name):
    os.makedirs(res_name)
model_name = os.path.join(res_name, 'model')
if not os.path.exists(model_name):
    os.makedirs(model_name)
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging_file_pth = os.path.join(res_name, 'training.log')
logging.basicConfig(filename=logging_file_pth, level=logging.CRITICAL, format=LOG_FORMAT, datefmt=DATE_FORMAT)
if not os.path.exists(res_name+'/code'):
    ignore_patterns = ['__pycache__', 'monai']
    shutil.copytree('../../code_TDNet', res_name+'/code', ignore=shutil.ignore_patterns(*ignore_patterns))


trans = {
    'train': al.Compose([
        al.RandomResizedCrop(height=320, width=200, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
        al.HorizontalFlip(p=0.1),
        al.VerticalFlip(p=0.1),
    ]),
    'test': al.Compose([
        al.Resize(height=320, width=200)
    ])
}


c16_32 = 32
dataset = air_dataset_2(args=args, train_path='../../train_data', test_path='../../test_data',
                        unlabel_path='../../unlabel_process', semi=False, transform=None, para=c16_32)
Loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0, pin_memory=False,
                                     collate_fn=collate_2)

with open('./configs/td2_psp50_cityscapes.yml') as fp:
    cfg = yaml.safe_load(fp)
teacher = get_model(cfg["teacher"], nclass=2) if 0 else None
net = get_model(cfg["model"], 2, [torch.nn.CrossEntropyLoss(), losses.DiceLoss(2)], cfg["training"]["resume"], teacher)

net.cuda()
net = torch.nn.DataParallel(net, device_ids=None)
print('-' * 10 + 'send to GPU' + '-' * 10)
optimizer = optim.Adam(net.parameters(), lr=0.001)

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    """
    example:
    >>> lr_scheduler(optimizer=optimizer, init_lr=args.lr, iter_num=None, max_iter=args.epochs*100, gamma=10, power=0.75)
    """
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
    return optimizer


# -----------------训练---------------------------------------
loss_all = []
iter_count = 0
best_score = 0.
best_epoch = 0
for epoch in range(args.epochs):
    running_loss = 0.0
    tr_loss = 0.0
    iters = len(Loader)
    iterator = iter(Loader)
    if args.earlystop:
        flag = ((epoch - best_epoch) > args.earlystop)
        if flag:
            print('early stop.')
            break
    for i in tqdm(range(iters), desc='training'):
        inputs, labels, _ = next(iterator)

        shifts_b_inputs = torch.roll(inputs, shifts=1, dims=0)
        shifts_b_inputs[0, ...] = 0.
        inp = torch.concat([shifts_b_inputs, inputs], dim=1)
        inp = inp.unsqueeze(dim=2).repeat(1, 1, 3, 1, 1)

        net.train()
        optimizer.zero_grad()
        lr_scheduler(optimizer=optimizer, init_lr=args.lr, iter_num=i,
                     max_iter=args.epochs * iters, gamma=10, power=0.75)

        p_0 = 0.95 * (epoch / args.epochs)
        p_1 = 1 - p_0
        pos_id = np.random.choice([0, 1], size=(1, ), replace=False, p=[p_0, p_1])[0]
        loss = net(inp, lbl=labels.squeeze(1), pos_id=int(pos_id))

        loss = gather(loss, 0)
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 0:
            logging.critical('epoch: {}, iter: {}, loss: {}'.format(epoch, i, running_loss / (i + 1)))
    tr_loss = running_loss / iters
    print('epoch_{:02d}'.format(epoch) + 'loss_{:f}'.format(tr_loss))

    if epoch % 1 == 0:
        net.eval()
        with torch.no_grad():
            dice = 0.
            count = 0
            for jj in tqdm(range(len(dataset.filenames_test)), desc='evaluating'):
                test_data = hdf.loadmat(os.path.join(dataset.test_path, dataset.filenames_test[jj]))['data_ima']
                test_data = test_data.transpose((2, 0, 1))
                test_data = np.expand_dims(test_data, axis=1)
                sli_num = test_data.shape[0]
                d__ = c16_32
                test_iteration = math.ceil(sli_num / d__)
                res = np.squeeze(np.zeros_like(test_data))
                for kk in range(test_iteration):
                    if kk != (test_iteration - 1):
                        in_data = torch.from_numpy(test_data[kk * d__:(kk + 1) * d__, :, :, 4:196]).float()
                    else:
                        assert kk == (test_iteration - 1)
                        in_data = torch.from_numpy(test_data[-d__:, :, :, 4:196]).float()
                    in_data = in_data.cuda()

                    shifts_b_inputs = torch.roll(in_data, shifts=1, dims=0)
                    shifts_b_inputs[0, ...] = 0.

                    in_data = torch.concat([shifts_b_inputs, in_data], dim=1)

                    in_data = in_data.unsqueeze(dim=2).repeat(1, 1, 3, 1, 1)

                    out_data = net(in_data, pos_id=kk%2)
                    _, predicted = torch.max(out_data, 1)
                    if kk != (test_iteration - 1):
                        res[kk * d__:(kk + 1) * d__, :, 4:196] = np.squeeze(predicted.cpu().detach().numpy())
                    else:
                        assert kk == (test_iteration - 1)
                        res[-d__:, :, 4:196] = np.squeeze(predicted.cpu().detach().numpy())

                res = res.transpose((1, 2, 0))
                test_lab = hdf.loadmat(os.path.join(dataset.test_path, dataset.filenames_test[jj]))['data_lab']
                dice += calculate_dice(test_lab, res[:, :, :sli_num])
                count += 1
            mean_dice = dice / count
            logging.critical('epoch: ' + str(epoch) + '_' + 'dice: ' + str(mean_dice))
            if mean_dice > best_score:
                best_score = mean_dice
                best_epoch = epoch
                # res_name_epoch = res_name+'/model_epoch_{:02d}'.format(epoch)
                # if not os.path.exists(res_name_epoch):
                #     os.makedirs(res_name_epoch)
                # sio.savemat(res_name_epoch+'/'+filenames_test[jj], {'res': res[:, :, :sli_num]})
                torch.save(net.state_dict(), model_name + '/model_epoch_{:02d}.pth'.format(epoch))
let_me_know("best_epoch: {}, best_score: {}".format(best_epoch, best_score))
print('All finish!')
