#!/usr/bin/env python
# pylint: disable=invalid-name, no-member
"""test Mini-Caffe"""
from __future__ import print_function
import os
import sys
import time
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(current_dir, '../')
model_dir = os.path.join(current_dir, '../../build/model')
sys.path.insert(0, lib_dir)
import minicaffe as mcaffe


if __name__ == '__main__':
    # check gpu available
    if mcaffe.check_gpu_available():
        mcaffe.set_runtime_mode(1, 0)
    # set up network
    net = mcaffe.Net(os.path.join(model_dir, 'resnet.prototxt'),
                     os.path.join(model_dir, 'resnet.caffemodel'))
    blob = net.get_blob('data')
    shape = blob.shape
    size = reduce(lambda acc, x: acc*x, shape, 1)
    blob.data[...] = np.random.rand(size).reshape(shape).astype(np.float32)
    # forward network
    t0 = time.clock()
    net.forward()
    t1 = time.clock()
    t = (t1 - t0) * 1000
    print('Forward ResNet costs %f ms'%t)
