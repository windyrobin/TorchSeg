#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.inferencer import Inferencer
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from tools.benchmark import compute_speed, stat
from datasets.TestData import TestData
#from datasets.cityscapes import Cityscapes
from datasets.etseg import ETSeg
from network import BiSeNet

logger = get_logger()


class SegInferencer(Inferencer):
    def func_per_iteration(self, data, device):
        img = data['data']
        name = data['fn']

        #img = cv2.resize(img, (config.image_width, config.image_height),
        #                 interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (config.image_width, config.image_height))

        pred = self.whole_eval(img,
                               #(config.image_height // config.gt_down_sampling,
                               # config.image_width // config.gt_down_sampling),
                               (config.image_height,
                                config.image_width),
                               device)

        if self.save_path is not None:
            colors = ETSeg.get_class_colors()
            image = img
            comp_img = show_img(colors, config.background, image,
                                pred)

            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), comp_img[:, :, ::-1])
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        #return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    parser.add_argument('--input_size', type=str, default='1x3x512x1024',
                        help='Input size. '
                             'channels x height x width (default: 1x3x224x224)')
    parser.add_argument('-speed', '--speed_test', action='store_true')
    parser.add_argument('--iteration', type=int, default=5000)
    parser.add_argument('-summary', '--summary', action='store_true')
    parser.add_argument('-trt', '--tensorrt', default=False, action='store_true')

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = BiSeNet(config.num_classes, is_training=False,
                      criterion=None, ohem_criterion=None)
    dataset = TestData('./fe_test')

    if args.speed_test:
        device = all_dev[0]
        logger.info("=========DEVICE:%s SIZE:%s=========" % (
            torch.cuda.get_device_name(device), args.input_size))
        input_size = tuple(int(x) for x in args.input_size.split('x'))
        compute_speed(network, input_size, device, args.iteration)
    elif args.summary:
        input_size = tuple(int(x) for x in args.input_size.split('x'))
        stat(network, input_size)
    else:
        with torch.no_grad():
            segmentor = SegInferencer(dataset, config.num_classes, config.image_mean,
                                     config.image_std, network,
                                     config.eval_scale_array, config.eval_flip,
                                     all_dev, args.verbose, args.save_path,
                                     args.show_image, args.tensorrt)
            segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                          config.link_val_log_file)
