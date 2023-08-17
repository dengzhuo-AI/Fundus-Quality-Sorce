import torch
import io
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from utils.registry import ARCH_REGISTRY
import softpool_cuda
from SoftPool import SoftPool2d
import pandas as pd
import torch,time
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as T
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from PIL import Image
from iqamodel_net import HyperSwin4, TargetNet

use_gpu = True

def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.
    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """

    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return image

def get_bare_model(net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

def model_to_device(net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net


def load_network(net, load_path, strict=True,param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)

image = Image.open('./sample_FQS/00058.jpeg')
image = prepare_image(image, target_size=(384, 384))
hyper = HyperSwin4(embed_dim=64,num_heads=[2, 4, 8, 16]).cuda()
load_network(hyper,'./pretrained/net_g_226264S4.pth')

hyper.eval()
with torch.no_grad():
    hy = hyper(image)
    model = TargetNet(hy).cuda()
    for param in model.parameters():
                param.requires_grad = False


    output = model(hy['target_in_vec'])

output = output.cpu().data
print(float(output))