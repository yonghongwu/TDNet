# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.ops
import torch.utils.checkpoint as checkpoint
from torch.distributions.uniform import Uniform


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder_DS(nn.Module):
    def __init__(self, params):
        super(Decoder_DS, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


class Decoder_URPC(nn.Module):
    def __init__(self, params):
        super(Decoder_URPC, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv_dp4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                      kernel_size=3, padding=1)
        self.out_conv_dp1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                      kernel_size=3, padding=1)
        self.feature_noise = FeatureNoise()

    def forward(self, feature, shape):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x = self.up1(x4, x3)
        if self.training:
            dp3_out_seg = self.out_conv_dp3(Dropout(x, p=0.5))
        else:
            dp3_out_seg = self.out_conv_dp3(x)
        dp3_out_seg = torch.nn.functional.interpolate(dp3_out_seg, shape)

        x = self.up2(x, x2)
        if self.training:
            dp2_out_seg = self.out_conv_dp2(FeatureDropout(x))
        else:
            dp2_out_seg = self.out_conv_dp2(x)
        dp2_out_seg = torch.nn.functional.interpolate(dp2_out_seg, shape)

        x = self.up3(x, x1)
        if self.training:
            dp1_out_seg = self.out_conv_dp1(self.feature_noise(x))
        else:
            dp1_out_seg = self.out_conv_dp1(x)
        dp1_out_seg = torch.nn.functional.interpolate(dp1_out_seg, shape)

        x = self.up4(x, x0)
        dp0_out_seg = self.out_conv(x)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


class UNet_CCT(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_CCT, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params)
        self.aux_decoder2 = Decoder(params)
        self.aux_decoder3 = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [FeatureNoise()(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        aux2_feature = [Dropout(i) for i in feature]
        aux_seg2 = self.aux_decoder2(aux2_feature)
        aux3_feature = [FeatureDropout(i) for i in feature]
        aux_seg3 = self.aux_decoder3(aux3_feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


class UNet_URPC(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_URPC, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_URPC(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg = self.decoder(
            feature, shape)
        return dp1_out_seg, dp2_out_seg, dp3_out_seg, dp4_out_seg


class UNet_DS(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_DS, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder_DS(params)

    def forward(self, x):
        shape = x.shape[2:]
        feature = self.encoder(x)
        dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg = self.decoder(
            feature, shape)
        return dp0_out_seg, dp1_out_seg, dp2_out_seg, dp3_out_seg


import sys

sys.path.append('../../')
from network_SwinUnet.swinV2 import WindowAttention
from einops import rearrange


class swin_attn_block(nn.Module):
    def __init__(self, dim, num_heads, window_size=(0, 0), qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        y = self.attn(x)
        return rearrange(y, 'b h w c -> b c h w')


class ConvBlock_2(ConvBlock):
    def __init__(self, in_channels, out_channels, dropout_p, use_attn=False, num_heads=1):
        super().__init__(in_channels, out_channels, dropout_p)
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
        )
        if use_attn:
            self.conv_conv_2 = nn.Sequential(
                swin_attn_block(dim=out_channels, num_heads=num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.1,
                                proj_drop=0.1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.conv_conv_2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        x1 = self.conv_conv(x)
        return self.conv_conv_2(x1) + x1


class DownBlock_2(DownBlock):
    def __init__(self, in_channels, out_channels, dropout_p, use_attn=False, num_heads=1):
        super().__init__(in_channels, out_channels, dropout_p)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock_2(in_channels, out_channels, dropout_p, use_attn=use_attn, num_heads=num_heads)
        )


class UpBlock_2(UpBlock):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True, use_attn=False, num_heads=1):
        super().__init__(in_channels1, in_channels2, out_channels, dropout_p, bilinear)
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock_2(in_channels2 * 2, out_channels, dropout_p, use_attn=use_attn, num_heads=num_heads)


class Encoder_2(Encoder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.num_heads = self.params['num_heads']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock_2(
            self.in_chns, self.ft_chns[0], self.dropout[0], use_attn=False)
        self.down1 = DownBlock_2(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1], use_attn=False, num_heads=self.num_heads[0])
        self.down2 = DownBlock_2(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2], use_attn=True, num_heads=self.num_heads[1])  # 提升了四倍
        self.down3 = DownBlock_2(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3], use_attn=True, num_heads=self.num_heads[2])
        self.down4 = DownBlock_2(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4], use_attn=True, num_heads=self.num_heads[3])


class Decoder_2(Decoder):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.num_heads = self.params['num_heads']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock_2(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear, use_attn=True,
            num_heads=self.num_heads[3])
        self.up2 = UpBlock_2(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear, use_attn=True,
            num_heads=self.num_heads[2])
        self.up3 = UpBlock_2(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear, use_attn=False,
            num_heads=self.num_heads[1])
        self.up4 = UpBlock_2(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear, use_attn=False,
            num_heads=self.num_heads[0])

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)


class FusionNet(UNet):
    def __init__(self, in_chns, class_num):
        super().__init__(in_chns, class_num)
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.2, 0.2],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu',
                  'num_heads': [2, 4, 8, 16]}

        self.encoder = Encoder_2(params)
        self.decoder = Decoder_2(params)

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output


# ---------3D----------#
#     structure       #
# ---------------------#
class ConvBlock_3D(ConvBlock):
    def __init__(self, in_channels, out_channels, dropout_p, is_spp=False):
        super().__init__(in_channels, out_channels, dropout_p)
        o1 = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, o1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(o1),
            nn.LeakyReLU(),
        )
        o2 = int(out_channels * 0.167)
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, o2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(o2),
            nn.LeakyReLU()
        )
        o3 = int(out_channels * 0.333)
        self.conv3 = nn.Sequential(
            nn.Conv3d(o2, o3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(o3),
            nn.LeakyReLU()
        )
        o4 = o1 - o2 - o3
        self.conv4 = nn.Sequential(
            nn.Conv3d(o3, o4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(o4),
            nn.LeakyReLU()
        )
        self.norm = nn.BatchNorm3d(out_channels)
        self.acti = nn.LeakyReLU()

    def forward(self, x):
        shortcut = self.conv1(x)
        out1 = self.conv2(x)
        out2 = self.conv3(out1)
        out3 = self.conv4(out2)
        c = torch.concat([out1, out2, out3], dim=1)
        return self.acti(self.norm(shortcut + c))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def autopad(k, p=None):  # kernel, padding
        # Pad to 'same'
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, e=0.5, k=(3, 5, 9)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool3d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class Res_path(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p, use_attn=False, num_heads=1, is_deform=False, repeat=1):
        super().__init__()
        # self.block = nn.Sequential(
        #     swin_attn_block_3D(dim=out_channels, num_heads=num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.1,
        #                        proj_drop=0.1, is_deform=is_deform),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU()
        )
        self.norm = nn.BatchNorm3d(out_channels)
        self.acti = nn.LeakyReLU()

        self.blocks = list([[self.conv1, self.conv2, self.norm, self.acti] for i in range(repeat)])

    def forward(self, x):
        for i, (conv1, conv2, norm, acti) in enumerate(self.blocks):
            if i == 0:
                x1 = conv1(x)
                x2 = conv2(x)
                x = acti(norm(x1 + x2))
            else:
                x1 = conv1(x)
                x2 = conv2(x)
                x = acti(norm(x1 + x2))
        return x


class UpBlock_3D(UpBlock):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, bilinear=True, up=True, is_spp=False):
        super().__init__(in_channels1, in_channels2, out_channels, dropout_p, bilinear)
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if up:
                self.up = nn.ConvTranspose3d(
                    in_channels1, in_channels2, kernel_size=4, stride=2, padding=1)
            else:
                self.up = nn.Conv3d(in_channels1, in_channels2, kernel_size=3, stride=1, padding=1)
        self.conv = ConvBlock_3D(in_channels2 * 2, out_channels, dropout_p, is_spp=is_spp)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class deform(nn.Module):
    def __init__(self, in_chns=3, out_chns=3, kernel_size=3, stride=1, padding=None):
        super().__init__()
        self.in_chns = in_chns
        self.out_chns = out_chns
        self.kernel_size = kernel_size
        self.padding = ((self.kernel_size - 1) + 1) // 2 if padding is None else 1
        self.conv = nn.Conv2d(self.in_chns, self.out_chns, kernel_size=self.kernel_size, stride=stride,
                              padding=self.padding)
        self.conv_offset = nn.Conv2d(self.in_chns, 2 * self.kernel_size * self.kernel_size, kernel_size=5, stride=1,
                                     padding=2)
        init_offset = torch.Tensor(np.zeros([2 * self.kernel_size * self.kernel_size, self.in_chns, 5, 5]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)
        self.conv_mask = nn.Conv2d(self.in_chns, self.kernel_size * self.kernel_size, kernel_size=5, stride=1,
                                   padding=2)
        init_mask = torch.Tensor(np.zeros([self.kernel_size * self.kernel_size, self.in_chns, 5, 5]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        return torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.conv.weight, mask=mask, padding=(1, 1))


class swin_attn_block_3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(0, 0), qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 is_deform=False):
        super().__init__()
        self.attn_hw = WindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.norm1 = nn.BatchNorm3d(dim)
        self.acti1 = nn.LeakyReLU()

        self.attn_hd = WindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.norm2 = nn.BatchNorm3d(dim)
        self.acti2 = nn.LeakyReLU()

        self.attn_dw = WindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.norm3 = nn.BatchNorm3d(dim)
        self.acti3 = nn.LeakyReLU()

        self.deform1 = deform(in_chns=dim, out_chns=dim) if is_deform else nn.Identity()
        self.deform2 = deform(in_chns=dim, out_chns=dim) if is_deform else nn.Identity()
        self.deform3 = deform(in_chns=dim, out_chns=dim) if is_deform else nn.Identity()

    def forward(self, x):
        b_, c_, d_, h_, w_ = x.shape
        x1 = rearrange(x, 'b c d h w -> (b d) c h w')
        x1 = self.deform1(x1)
        x1 = rearrange(x1, '(b d) c h w -> (b d) h w c', d=d_)
        y1 = self.attn_hw(x1.contiguous())
        y1 = rearrange(y1, '(b d) h w c -> b c d h w', d=d_)

        x2 = rearrange(x, 'b c d h w -> (b w) c h d')
        x2 = self.deform2(x2)
        x2 = rearrange(x2, '(b w) c h d -> (b w) h d c', w=w_)
        y2 = self.attn_hd(x2.contiguous())
        y2 = rearrange(y2, '(b w) h d c -> b c d h w', w=w_)

        x3 = rearrange(x, 'b c d h w -> (b h) c d w')
        x3 = self.deform3(x3)
        x3 = rearrange(x3, '(b h) c d w -> (b h) d w c', h=h_)
        y3 = self.attn_dw(x3.contiguous())
        y3 = rearrange(y3, '(b h) d w c -> b c d h w', h=h_)

        return self.acti1(self.norm1(y1)) + self.acti2(self.norm2(y2)) + self.acti3(self.norm3(y3))


class Encoder_3D(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        self.num_heads = self.params['num_heads']
        self.conv1 = ConvBlock_3D(self.in_chns, self.ft_chns[0], self.dropout[0], is_spp=False)
        self.conv2 = ConvBlock_3D(self.ft_chns[0], self.ft_chns[1], self.dropout[1], is_spp=False)
        self.conv3 = ConvBlock_3D(self.ft_chns[1], self.ft_chns[2], self.dropout[2], is_spp=False)
        self.conv4 = ConvBlock_3D(self.ft_chns[2], self.ft_chns[3], self.dropout[3], is_spp=False)

        self.res1 = Res_path(self.ft_chns[0], self.ft_chns[0], 0., repeat=4)
        self.res2 = Res_path(self.ft_chns[1], self.ft_chns[1], 0., repeat=3)
        self.res3 = Res_path(self.ft_chns[2], self.ft_chns[2], 0., repeat=2)
        self.res4 = Res_path(self.ft_chns[3], self.ft_chns[3], 0., repeat=1)

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        z1 = self.res1(x1)   # (16, 320, 192)
        x1 = self.maxpool(x1)

        x1 = self.conv2(x1)
        z2 = self.res2(x1)   # (8, 160, 96)
        x1 = self.maxpool(x1)

        x1 = self.conv3(x1)
        z3 = self.res3(x1)   # (4, 80, 48)
        x1 = self.maxpool(x1)

        x1 = self.conv4(x1)
        z4 = self.res4(x1)   # (2, 40, 24)
        x1 = self.maxpool(x1)

        return x1, [z1, z2, z3, z4]


class Decoder_3D(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.num_heads = self.params['num_heads']

        self.up1 = UpBlock_3D(self.ft_chns[3], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0, bilinear=self.bilinear,
                              up=True)
        self.up2 = UpBlock_3D(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0, bilinear=self.bilinear,
                              is_spp=False)
        self.up3 = UpBlock_3D(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0, bilinear=self.bilinear,
                              is_spp=False)
        self.up4 = UpBlock_3D(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0, bilinear=self.bilinear,
                              is_spp=False)

        self.out_conv = ConvBlock_3D(in_channels=self.ft_chns[0], out_channels=self.ft_chns[0], dropout_p=0.0)

    def forward(self, feature, connections):
        x = self.up1(feature, connections[3])
        x = self.up2(x, connections[2])
        x = self.up3(x, connections[1])
        x = self.up4(x, connections[0])

        output = self.out_conv(x)
        return output


class FusionNet_3D(UNet):
    def __init__(self, in_chns, class_num):
        super().__init__(in_chns, class_num)
        params = {'in_chns': in_chns,
                  'feature_chns': [10, 20, 40, 80],
                  'dropout': [0.05, 0.1, 0.2, 0.2, 0.2],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'leakyrelu',
                  'num_heads': [1, 2, 4, 8]}

        self.encoder = Encoder_3D(params)
        self.bottom = ConvBlock_3D(in_channels=params['feature_chns'][-1], out_channels=params['feature_chns'][-1], dropout_p=0.)
        self.decoder = Decoder_3D(params)
        self.out_conv = nn.Conv3d(in_channels=params['feature_chns'][0], out_channels=params['class_num'], kernel_size=(1, 1, 1), stride=(1, 1, 1), padding='same')

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        feature, connections = self.encoder(x)
        bottom = self.bottom(feature)
        output = self.decoder(bottom, connections)
        output = self.out_conv(output)
        return output


if __name__ == '__main__':
    import os
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    class parser:
        def __init__(self):
            pass


    args = parser()
    args.batchsize = 3

    model = FusionNet(in_chns=1, class_num=2)
    params = 0
    for name, para in model.named_parameters():
        params += torch.numel(para)
    x = torch.randn(size=(16 * args.batchsize, 1, 320, 192))
    t1 = time.time()
    y = model(x)
    t2 = time.time()
    print('finish>>> Time cost: {}'.format(t2 - t1))

    model_3D = FusionNet_3D(in_chns=1, class_num=2)
    params = 0
    for name, para in model_3D.named_parameters():
        params += torch.numel(para)
    for name, module in model_3D.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    x = torch.randn(size=(args.batchsize, 1, 16, 320, 192))
    t1 = time.time()
    y = model_3D(x)
    t2 = time.time()
    print('finish>>> Time cost: {}'.format(t2 - t1))
