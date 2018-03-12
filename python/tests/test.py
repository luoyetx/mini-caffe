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


def test_crafter():
    """test crafter"""
    Crafter = mcaffe.LayerCrafter
    conv = Crafter(name='conv1',
                   bottom=['in1', 'in2'],
                   top=['out1', 'out2'],
                   type='Convolution',
                   convolution_param={
                       'num_output': 20,
                       'kernel_size': 4,
                       'weight_filler': {
                           'type': 'xavier',
                       }
                   },
                   dummy_test1=[1., 2., 3.],
                   dummy_test2=True).gen()
    print(conv)


def test_network():
    """test network"""
    # check gpu available
    if mcaffe.check_gpu_available():
        mcaffe.set_runtime_mode(1, 0)
    mcaffe.Profiler.enable()
    # set up network
    net = mcaffe.Net(os.path.join(model_dir, 'resnet.prototxt'),
                     os.path.join(model_dir, 'resnet.caffemodel'))
    net.mark_output("conv1")
    mcaffe.Profiler.open_scope("resnet")
    blob = net.get_blob('data')
    shape = blob.shape
    size = 1
    for s in shape:
        size *= s
    blob.data[...] = np.random.rand(size).reshape(shape).astype(np.float32)
    # forward network
    t0 = time.clock()
    net.forward()
    t1 = time.clock()
    t = (t1 - t0) * 1000
    mcaffe.Profiler.close_scope()
    mcaffe.Profiler.disable()
    mcaffe.Profiler.dump("resnet-profile.json")
    print('Forward ResNet costs %f ms'%t)
    # forward network by pass data
    net.forward(**{'data': np.random.rand(size).reshape(shape).astype(np.float32)})
    # network parameters
    params = net.params
    for layer_name, layer_params in list(params.items()):
        print('layer: {\n\tname: %s'%layer_name)
        for name, param in layer_params:
            shape = param.shape
            print('\t{}: {}'.format(name, shape))
        print('}')
    # network internal blobs
    blobs = net.blobs
    print('{')
    for name, blob in list(blobs.items()):
        shape = blob.shape
        print('\t{}: {}'.format(name, shape))
    print('}')


if __name__ == '__main__':
    # test crafter
    test_crafter()
    test_network()
