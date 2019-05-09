#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch

from network import BiSeNet
from config import config

from tensorboardX import SummaryWriter

network = BiSeNet(config.num_classes, is_training=False,
              criterion=None, ohem_criterion=None)

dummy_input = torch.randn(1, 3, 768, 768*2)

with SummaryWriter(comment="bisenet") as w:
    w.add_graph(network, (dummy_input, ))
