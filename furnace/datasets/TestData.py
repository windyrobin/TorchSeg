#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import cv2
import torch
import numpy as np

import torch.utils.data as data


class TestData(data.Dataset):
    def __init__(self, img_path, preprocess=None,
                 file_length=None):
        super(TestData, self).__init__()
        self._img_path = img_path
        self._file_names = self._get_file_names(img_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self._file_names)

    def __getitem__(self, index):
        name = self._file_names[index]
        img_path = os.path.join(self._img_path, name)
        item_name = name.split("/")[-1].split(".")[0]

        img= self._fetch_data(img_path)
        img = img[:, :, ::-1]

        output_dict = dict(data=img, fn=str(item_name),
                           n=len(self._file_names))
        return output_dict

    def _fetch_data(self, img_path, dtype=None):
        img = self._open_image(img_path)
        return img
    #TODO, list dir
    def _get_file_names(self, source):

        file_names = os.listdir(source)
        #with open(source) as f:
        #    files = f.readlines()

        #for item in files:
        #    img_name = self._process_item_names(item)
        #    file_names.append(img_name)
        return file_names
        #return file_names[:1]

    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        return item

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)

        return img

    @classmethod
    def get_class_colors(*args):
        raise NotImplementedError

    @classmethod
    def get_class_names(*args):
        raise NotImplementedError


if __name__ == "__main__":
    data_setting = {'img_root': ''}
    bd = TestData(data_setting, 'test/', None)
    print(bd.get_class_names())
