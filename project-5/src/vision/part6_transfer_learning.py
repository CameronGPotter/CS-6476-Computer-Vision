import logging
import os
import pdb
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from src.vision.part5_pspnet import PSPNet
from src.vision.utils import load_class_names, get_imagenet_mean_std, get_logger, normalize_img


_ROOT = Path(__file__).resolve().parent.parent.parent

logger = get_logger()


def load_pretrained_model(args, use_cuda: bool):
    """Load Pytorch pre-trained PSPNet model from disk of type torch.nn.DataParallel.

    Note that `args.num_model_classes` will be size of logits output.

    Args:
        args:
        use_cuda:

    Returns:
        model
    """
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = PSPNet(
        layers=args.layers,
        num_classes=args.classes,
        zoom_factor=args.zoom_factor,
        criterion=criterion,
        pretrained=False
    )

    # logger.info(model)
    if use_cuda:
        model = model.cuda()
    cudnn.benchmark = True

    if os.path.isfile(args.model_path):
        logger.info(f"=> loading checkpoint '{args.model_path}'")
        if use_cuda:
            checkpoint = torch.load(args.model_path)
        else:
            checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        logger.info(f"=> loaded checkpoint '{args.model_path}'")
    else:
        raise RuntimeError(f"=> no checkpoint found at '{args.model_path}'")

    return model



def model_and_optimizer(args, model) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """
    This function is similar to get_model_and_optimizer in Part 3.

    Use the model trained on Camvid as the pretrained PSPNet model, change the
    output classes number to 2 (the number of classes for Kitti).
    Refer to Part 3 for optimizer initialization.

    Args:
        args: object containing specified hyperparameters
        model: pre-trained model on Camvid

    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    new_model = PSPNet(
        layers=args.layers,
        num_classes=2,
        zoom_factor=args.zoom_factor,
        pretrained=args.pretrained
    )

    new_model.resnet = model
    model = new_model
    
    params = [
        {'params': model.layer0.parameters(), 'lr': args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}, 
        {'params': model.layer1.parameters(), 'lr': args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}, 
        {'params': model.layer2.parameters(), 'lr': args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}, 
        {'params': model.layer3.parameters(), 'lr': args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}, 
        {'params': model.layer4.parameters(), 'lr': args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay}
    ]
    if model.use_ppm:
        params.append({'params': model.ppm.parameters(), 'lr': 10*args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay})
    params.append({'params': model.cls.parameters(), 'lr': 10*args.base_lr, 'momentum': args.momentum, 'weight_decay': args.weight_decay})

    optimizer = torch.optim.SGD(params)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return model, optimizer
