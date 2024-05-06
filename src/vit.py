

import os
import sys
import time
import math
# import argparse
from random import randrange
from functools import partial

import numpy as np
import random
import pandas as pd
import csv
import time

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

import torch
from torch import einsum
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torchinfo import summary

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw


# ----------
# REFERENCE:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch
# https://github.com/kentaroy47/vision-transformers-cifar10/tree/main


base_path = '~/vision_transformers'
data_dir = os.path.join(base_path, '01_data')
out_dir = os.path.join(base_path, '04_output')


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# helper functions
# ----------------------------------------------------------------------------------------------------------------

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant(m.weight, 1)
            torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                torch.nn.init.constant(m.bias, 0)


# ----------
try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = 80

term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


###################################################################################################
# -------------------------------------------------------------------------------------------------
# functions for data augmentation
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# -------------------------------------------------------------------------------------------------


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype('int')
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = PIL.Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


###################################################################################################
# -------------------------------------------------------------------------------------------------
# VIT
# -------------------------------------------------------------------------------------------------

def pair_vit(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward_vit(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            # Layer Norm is applied first !!!
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention_vit(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # Layer Norm is applied first !!!
        x = self.norm(x)

        # ----------
        # self-attention (all q, k, v comes from same x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_vit(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_vit(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward_vit(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        # Layer Norm is applied
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair_vit(image_size)
        patch_height, patch_width = pair_vit(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # Layer Norm is applied
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # Layer Norm is applied
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_vit(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # ----------
        # Linear Projection of Flattened Patches
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # ----------
        # Patch + Position Embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # ----------
        # Transformer Encoder
        x = self.transformer(x)

        # ----------
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


# -------------------------------------------------------------------------------------------------
# for comparison ResNet18
# -------------------------------------------------------------------------------------------------

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class BasicBlock_resnet(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_resnet(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(**kwargs):
    return ResNet(BasicBlock_resnet, [2, 2, 2, 2])

def ResNet34(**kwargs):
    return ResNet(BasicBlock_resnet, [3, 4, 6, 3])

def ResNet50(**kwargs):
    return ResNet(Bottleneck_resnet, [3, 4, 6, 3])

def ResNet101(**kwargs):
    return ResNet(Bottleneck_resnet, [3, 4, 23, 3])

def ResNet152(**kwargs):
    return ResNet(Bottleneck_resnet, [3, 8, 36, 3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())


# -------------------------------------------------------------------------------------------------
# Swin Transformer
# -------------------------------------------------------------------------------------------------

# https://github.com/berniwal/swin-transformer-pytorch

class CyclicShift_swt(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual_swt(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm_swt(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward_swt(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask_swt(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances_swt(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention_swt(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift_swt(-displacement)
            self.cyclic_back_shift = CyclicShift_swt(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask_swt(
                window_size=window_size, displacement=displacement,
                upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask_swt(
                window_size=window_size, displacement=displacement,
                upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances_swt(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual_swt(
            PreNorm_swt(
                dim, WindowAttention_swt(
                    dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding
                )))
        self.mlp_block = Residual_swt(
            PreNorm_swt(
                dim, FeedForward_swt(
                    dim=dim, hidden_dim=mlp_dim
                )))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x


class PatchMerging_swt(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class StageModule_swt(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging_swt(
            in_channels=in_channels, out_channels=hidden_dimension,
            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule_swt(
            in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
            downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
            window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule_swt(
            in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
            downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
            window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule_swt(
            in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
            downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
            window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule_swt(
            in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
            downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
            window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


# -------------------------------------------------------------------------------------------------
# CaiT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cait.py
# -------------------------------------------------------------------------------------------------

def exists_cait(val):
    return val is not None

def dropout_layers_cait(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


class LayerScale_cait(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm_cait(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward_cait(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention_cait(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = nn.Softmax(dim=-1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists_cait(context) else torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer_cait(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., layer_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale_cait(dim, PreNorm_cait(dim, Attention_cait(dim, heads=heads, dim_head=dim_head, dropout=dropout)), depth=ind + 1),
                LayerScale_cait(dim, PreNorm_cait(dim, FeedForward_cait(dim, mlp_dim, dropout=dropout)), depth=ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers_cait(self.layers, dropout=self.layer_dropout)

        for attn, ff in layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        return x


class CaiT(nn.Module):
    def __init__(self, *,
                 image_size, patch_size, num_classes, dim, depth, cls_depth,
                 heads, mlp_dim,
                 dim_head=64, dropout=0., emb_dropout=0., layer_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer_cait(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        self.cls_transformer = Transformer_cait(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = self.cls_transformer(cls_tokens, context=x)

        return self.mlp_head(x[:, 0])


# -------------------------------------------------------------------------------------------------
# convmixer
# https://openreview.net/forum?id=TVHS5Y4dNvM
# -------------------------------------------------------------------------------------------------

class Residual_convmixer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=9, patch_size=7, num_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual_convmixer(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes)
    )


# -------------------------------------------------------------------------------------------------
# MLPMixer
# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
# -------------------------------------------------------------------------------------------------

pair_mlpmixer = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual_mlpmixer(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward_mlpmixer(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes,
             expansion_factor=4, expansion_factor_token=0.5, dropout=0.):
    image_h, image_w = pair_mlpmixer(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual_mlpmixer(dim, FeedForward_mlpmixer(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual_mlpmixer(dim, FeedForward_mlpmixer(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


###################################################################################################
# -------------------------------------------------------------------------------------------------
# train
# -------------------------------------------------------------------------------------------------

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return train_loss / (batch_idx + 1)


def test(epoch, log_path, checkpoint_path):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        torch.save(state, checkpoint_path)
        best_acc = acc

    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(log_path, 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc


###################################################################################################
# -------------------------------------------------------------------------------------------------
# prepare dataset and dataloader
# -------------------------------------------------------------------------------------------------

SEED = 3497
reproducibility(SEED)

image_size = 32

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


batch_size = 512

trainset = torchvision.datasets.CIFAR10(root='./01_data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./01_data', train=False, download=True, transform=transform_test)

# trainset = torchvision.datasets.CIFAR100(root='./01_data', train=True, download=True, transform=transform_train)
# testset = torchvision.datasets.CIFAR100(root='./01_data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# classes = [
#     'apples',  # 0：りんご
#     'aquarium fish',  # 1：観賞魚
#     'baby',  # 2：赤ちゃん
#     'bear',  # 3：クマ
#     'beaver',  # 4：ビーバー
#     'bed',  # 5：ベッド
#     'bee',  # 6：蜂
#     'beetle',  # 7：カブトムシ
#     'bicycle',  # 8：自転車
#     'bottles',  # 9：ボトル
#     'bowls',  # 10：ボウル
#     'boy',  # 11：少年
#     'bridge',  # 12：橋
#     'bus',  # 13：バス
#     'butterfly',  # 14：蝶
#     'camel',  # 15：ラクダ
#     'cans',  # 16：缶
#     'castle',  # 17：城
#     'caterpillar',  # 18：毛虫
#     'cattle',  # 19：牛
#     'chair',  # 20：椅子
#     'chimpanzee',  # 21：チンパンジー
#     'clock',  # 22：時計
#     'cloud',  # 23：雲
#     'cockroach',  # 24：ゴキブリ
#     'couch',  # 25：ソファー
#     'crab',  # 26：カニ
#     'crocodile',  # 27：ワニ
#     'cups',  # 28：カップ
#     'dinosaur',  # 29：恐竜
#     'dolphin',  # 30：イルカ
#     'elephant',  # 31：象
#     'flatfish',  # 32：ヒラメ
#     'forest',  # 33：森
#     'fox',  # 34：キツネ
#     'girl',  # 35：少女
#     'hamster',  # 36：ハムスター
#     'house',  # 37：家
#     'kangaroo',  # 38：カンガルー
#     'computer keyboard',  # 39：コンピューターのキーボード
#     'lamp',  # 40：ランプ
#     'lawn-mower',  # 41：芝刈り機
#     'leopard',  # 42：ヒョウ
#     'lion',  # 43：ライオン
#     'lizard',  # 44：トカゲ
#     'lobster',  # 45：ロブスター
#     'man',  # 46：成人男性
#     'maple',  # 47：もみじ
#     'motorcycle',  # 48：オートバイ
#     'mountain',  # 49：山
#     'mouse',  # 50：ねずみ
#     'mushrooms',  # 51：きのこ
#     'oak',  # 52：オーク
#     'oranges',  # 53：オレンジ
#     'orchids',  # 54：蘭
#     'otter',  # 55：カワウソ
#     'palm',  # 56：ヤシ
#     'pears',  # 57：洋ナシ
#     'pickup truck',  # 58：ピックアップトラック
#     'pine',  # 59：松
#     'plain',  # 60：平野
#     'plates',  # 61：皿
#     'poppies',  # 62：ポピー
#     'porcupine',  # 63：ヤマアラシ
#     'possum',  # 64：フクロネズミ
#     'rabbit',  # 65：ウサギ
#     'raccoon',  # 66：アライグマ
#     'ray',  # 67：エイ
#     'road',  # 68：道路
#     'rocket',  # 69：ロケット
#     'roses',  # 70：バラ
#     'sea',  # 71：海
#     'seal',  # 72：アザラシ
#     'shark',  # 73：サメ
#     'shrew',  # 74：トガリネズミ
#     'skunk',  # 75：スカンク
#     'skyscraper',  # 76：超高層ビル
#     'snail',  # 77：カタツムリ
#     'snake',  # 78：ヘビ
#     'spider',  # 79：クモ
#     'squirrel',  # 80：リス
#     'streetcar',  # 81：路面電車
#     'sunflowers',  # 82：ひまわり
#     'sweet peppers',  # 83：パプリカ
#     'table',  # 84：テーブル
#     'tank',  # 85：タンク
#     'telephone',  # 86：電話
#     'television',  # 87：テレビ
#     'tiger',  # 88：トラ
#     'tractor',  # 89：トラクター
#     'train',  # 90：電車
#     'trout',  # 91：マス
#     'tulips',  # 92：チューリップ
#     'turtle',  # 93：カメ
#     'wardrobe',  # 94：ワードローブ
#     'whale',  # 95：クジラ
#     'willow',  # 96：柳
#     'wolf',  # 97：オオカミ
#     'woman',  # 98：成人女性
#     'worm',  # 99：ミミズ
# ]


num_classes = len(classes)


# ----------
N = 2
M = 14
transform_train.transforms.insert(0, RandAugment(N, M))


###################################################################################################
# -------------------------------------------------------------------------------------------------
# common settings
# -------------------------------------------------------------------------------------------------

device = 'cuda'

lr = 1e-4
n_epochs = 200
use_amp = True

usewandb = True


###################################################################################################
# -------------------------------------------------------------------------------------------------
# model
# -------------------------------------------------------------------------------------------------

# # Vision Transformer base model (from paper)
# # Base:  layer 12  hidden size 768  heads 12  -->  Param 86M
#
# args_vit = {
#     'project_name': 'cifar10-challenge_VIT',
#     'net_name': 'VIT',
#     'SEED': SEED,
#     'image_size': image_size,
#     'patch_size': 4,
#     'num_classes': num_classes,
#     'dimhead': 512,  # (image_size / patch_size)^2 * heads = num_patches * heads = 64 * 8 = 512
#     'depth': 6,
#     'cls_depth': np.nan,
#     'heads': 8,
#     'mlp_dim': 512,
#     'channels': np.nan,
#     'dropout': 0.1,
#     'emb_dropout': 0.1,
#     'layer_dropout': np.nan,
#     'convkernel': np.nan,
#     'use_ump': use_amp,
#     # 'mixup': mixup,
#     'batch_size': batch_size,
#     'lr': lr,
#     'n_epochs': n_epochs,
# }
#
# net_vit = ViT(
#     image_size=args_vit['image_size'],
#     patch_size=args_vit['patch_size'],
#     num_classes=num_classes,
#     dim=int(args_vit['dimhead']),
#     depth=int(args_vit['depth']),
#     heads=int(args_vit['heads']),
#     mlp_dim=int(args_vit['mlp_dim']),
#     dropout=args_vit['dropout'],
#     emb_dropout=args_vit['emb_dropout'],
# )
#
# # here only 9.52M params
# print(summary(net_vit))
#
#
# # ----------
# # for comparison ResNet18
# args_resnet18 = {
#     'project_name': 'cifar10-challenge_ResNet18',
#     'net_name': 'ResNet18',
#     'SEED': SEED,
#     'image_size': np.nan,
#     'patch_size': np.nan,
#     'num_classes': num_classes,
#     'dimhead': np.nan,
#     'depth': np.nan,
#     'cls_depth': np.nan,
#     'heads': np.nan,
#     'mlp_dim': np.nan,
#     'channels': np.nan,
#     'dropout': np.nan,
#     'emb_dropout': np.nan,
#     'layer_dropout': np.nan,
#     'convkernel': np.nan,
#     'use_ump': use_amp,
#     # 'mixup': mixup,
#     'batch_size': batch_size,
#     'lr': lr,
#     'n_epochs': n_epochs,
# }
#
# net_resnet18 = ResNet18(
#     num_classes=args_resnet18['num_classes'],
# )
#
# # 11.17M params
# print(summary(net_resnet18))
#
#
# # ----------
# # CaiT
# args_cait = {
#     'project_name': 'cifar10-challenge_CaiT',
#     'net_name': 'CaiT',
#     'SEED': SEED,
#     'image_size': image_size,
#     'patch_size': 4,
#     'num_classes': num_classes,
#     'dimhead': 512,
#     'depth': 6, # depth of transformer for patch to patch attention only
#     'cls_depth': 2, # depth of cross attention of CLS tokens to patch
#     'heads': 8,
#     'mlp_dim': 512,
#     'channels': np.nan,
#     'dropout': 0.1,
#     'emb_dropout': 0.1,
#     'layer_dropout': 0.05,
#     'convkernel': np.nan,
#     'use_ump': use_amp,
#     # 'mixup': mixup,
#     'batch_size': batch_size,
#     'lr': lr,
#     'n_epochs': n_epochs,
# }
#
# net_cait = CaiT(
#     image_size=args_cait['image_size'],
#     patch_size=args_cait['patch_size'],
#     num_classes=num_classes,
#     dim=int(args_cait['dimhead']),
#     depth=int(args_cait['depth']),
#     cls_depth=int(args_cait['cls_depth']),
#     heads=int(args_cait['heads']),
#     mlp_dim=int(args_cait['mlp_dim']),
#     dropout=args_cait['dropout'],
#     emb_dropout=args_cait['emb_dropout'],
#     layer_dropout=args_cait['layer_dropout'],
# )
#
# # here 12.66M params
# print(summary(net_cait))
#
#
# # ----------
# # Swin Transformers
# args_swt = {
#     'project_name': 'cifar10-challenge_SwinT',
#     'net_name': 'SwinT',
#     'SEED': SEED,
#     'image_size': np.nan,
#     'patch_size': 4,
#     'num_classes': num_classes,
#     'dimhead': 512,
#     'depth': np.nan,
#     'cls_depth': np.nan,
#     'heads': np.nan,
#     'channels': np.nan,
#     'mlp_dim': np.nan,
#     'dropout': np.nan,
#     'emb_dropout': np.nan,
#     'layer_dropout': np.nan,
#     'convkernel': np.nan,
#     'use_ump': use_amp,
#     # 'mixup': mixup,
#     'batch_size': batch_size,
#     'lr': lr,
#     'n_epochs': n_epochs,
# }
#
# net_swint = swin_t(
#     window_size=args_swt['patch_size'],
#     num_classes=args_swt['num_classes'],
#     downscaling_factors=(2, 2, 2, 1))
#
# # here 26.60M params .. large
# print(summary(net_swint))


# ----------
# ConvMixer
# from paper, accuracy > 96%.
# you can tune the depth and dim to scale accuracy and speed.
args_convmixer = {
    'project_name': 'cifar10-challenge_ConvMixer',
    'net_name': 'ConvMixer',
    'SEED': SEED,
    'image_size': np.nan,
    'patch_size': 1,
    'num_classes': num_classes,
    'dimhead': np.nan,
    'depth': 16,
    'cls_depth': np.nan,
    'heads': np.nan,
    'mlp_dim': 256,
    'channels': np.nan,
    'dropout': np.nan,
    'emb_dropout': np.nan,
    'layer_dropout': np.nan,
    'convkernel': 8,
    'use_ump': use_amp,
    # 'mixup': mixup,
    'batch_size': batch_size,
    'lr': lr,
    'n_epochs': n_epochs,
}

net_convmixer = ConvMixer(
    dim=args_convmixer['mlp_dim'],
    depth=args_convmixer['depth'],
    kernel_size=args_convmixer['convkernel'],
    patch_size=args_convmixer['patch_size'],
    num_classes=args_convmixer['num_classes']
)

# 1.34M params !!
print(summary(net_convmixer))


# ----------
# MLPMixer
args_mlpmixer = {
    # 'project_name': 'cifar10-challenge_MLPMixer',
    'project_name': 'cifar10-challenge_MLPMixer4',
    'net_name': 'MLPMixer',
    'SEED': SEED,
    'image_size': image_size,
    'patch_size': 4,
    'num_classes': num_classes,
    'dimhead': np.nan,
    'depth': 6,
    'cls_depth': np.nan,
    'heads': np.nan,
    'mlp_dim': 512,
    'channels': 3,
    'dropout': np.nan,
    'emb_dropout': np.nan,
    'layer_dropout': np.nan,
    'convkernel': np.nan,
    'use_ump': use_amp,
    # 'mixup': mixup,
    'batch_size': batch_size,
    'lr': lr,
    'n_epochs': n_epochs,
}

net_mlpmixer = MLPMixer(
    image_size=args_mlpmixer['image_size'],
    channels=args_mlpmixer['channels'],
    patch_size=args_mlpmixer['patch_size'],
    dim=args_mlpmixer['mlp_dim'],
    depth=args_mlpmixer['depth'],
    num_classes=args_mlpmixer['num_classes'],
)

# 1.82M params !!
print(summary(net_mlpmixer))


###################################################################################################
# -------------------------------------------------------------------------------------------------
# training by each model
# -------------------------------------------------------------------------------------------------

net_list = [
    # net_resnet18,
    # net_vit,
    # net_cait,
    # net_convmixer,
    net_mlpmixer,
]

args_list = [
    # args_resnet18,
    # args_vit,
    # args_cait,
    # args_convmixer,
    args_mlpmixer,
]

for net, args_dict in zip(net_list, args_list):

    reproducibility(SEED)

    net_name = args_dict['net_name']
    patch_size = args_dict['patch_size']
    project_name = args_dict['project_name']

    if usewandb:
        import wandb
        watermark = f"{net_name}_lr{lr}"
        wandb.init(project=project_name, name=watermark)
        wandb.config.update(args_dict, allow_val_change=True)

    acc_log_path = os.path.join(base_path, '04_output', 'log', f'acc_{net_name}_patch{patch_size}.csv')
    log_path = os.path.join(base_path, '04_output', 'log', f'log_{net_name}_patch{patch_size}.txt')
    checkpoint_path = os.path.join(base_path, '04_output', 'checkpoint', f'checkpoint_{net_name}_patch{patch_size}.ckpt.t7')

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    # ----------
    # Adam is better at high start !!
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # ----------
    # use cosine scheduling
    # ReduceLROnPlateau is difficult to control
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=20)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.95, patience=40)

    # ----------
    # To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and
    # invokes a backward pass on the scaled loss(es).
    # Gradients flowing backward through the network are then scaled by the same factor.
    # In other words, gradient values have a larger magnitude, so they don’t flush to zero.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    net.cuda()
    start_epoch = 0

    if usewandb:
        wandb.watch(net)

    list_loss = []
    list_acc = []
    best_acc = 0

    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc = test(epoch, log_path, checkpoint_path)

        scheduler.step(epoch - 1)

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Log training..
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc,
                       "lr": optimizer.param_groups[0]["lr"],
                       "epoch_time": time.time() - start})

        # Write out csv
        with open(acc_log_path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)

    # does not work ...
    # wandb_save_fname = f'wandb_{net_name}_patch{patch_size}.h5'
    # wandb_save_basepath = os.path.join(base_path, '04_output', 'wandb')
    # if usewandb:
    #     wandb.save(path=wandb_save_fname, base_path=wandb_save_basepath)

    wandb.finish()

