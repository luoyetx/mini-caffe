# coding: utf-8
# pylint: disable=invalid-name, no-member, line-too-long

from __future__ import print_function

import os
import argparse
import numpy as np
import caffe_pb2
from google.protobuf import text_format


def fuse(weight, bias, mean, var, scale, shift, eps=1e-5):
    """fuse conv weight with bn parameters
    Parameters
    ==========
    weight, bias: np.array
        convolution weight and bias
        weight: [C_out, C_in, K_h, K_w]
        bias: [C_out]
    mean, var, scale, shift: np.array
        bn parameters
        mean, var, scale, shift: [C_out]
    eps: float
        eps

    Returns
    =======
    weight, bias: np.array
        convolution weight and bias
        weight: [C_out, C_in, K_h, K_w]
        bias: [C_out]
    """
    std = np.sqrt(var + eps)
    weight_shape = weight.shape
    weight = weight.reshape((weight_shape[0], -1)) * scale.reshape((-1, 1)) / std.reshape((-1, 1))
    weight = weight.reshape(weight_shape)
    bias = (bias - mean) * scale / std + shift
    return weight, bias


def get_layer(net, layer_name):
    """get layer object by layer name
    Parameters
    ==========
    net: NetParameter
        network
    layer_name: str
        layer name

    Returns
    =======
    layer: LayerParameter or None
        layer, None if not found
    """
    for layer in net.layer:
        if layer.name == layer_name:
            return layer
    return None


def convert_blob_to_array(blob):
    """convert caffe blob to numpy array
    Parameters
    ==========
    blob: caffe.BlobProto
        blob

    Returns
    =======
    arr: np.array
        array
    """
    shape = [d for d in blob.shape.dim]
    arr = np.array(blob.data, dtype=np.float32)
    arr = arr.reshape(shape)
    return arr


def convert_array_to_blob(arr):
    """convert numpy array to caffe blob
    Parameters
    ==========
    arr: np.array
        array

    Returns
    =======
    blob: caffe.BlobProto
        blob
    """
    blob = caffe_pb2.BlobProto()
    blob.shape.dim.extend(arr.shape)
    blob.data.extend(arr.flatten())
    return blob


def main():
    """main
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='net prototxt')
    parser.add_argument('--weight', type=str, required=True, help='net weight')
    args = parser.parse_args()
    print(args)

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.net, 'r').read(), net)
    weight = caffe_pb2.NetParameter()
    weight.ParseFromString(open(args.weight, 'rb').read())

    # remove useless layers
    net_layers = [layer.name for layer in net.layer]
    weight_layers = [layer.name for layer in net.layer]
    not_used_layers = [layer for layer in weight_layers if layer not in net_layers]

    for layer_name in not_used_layers:
        weight.layer.remove(get_layer(weight, layer_name))

    # search conv-batchnorm-scale pattern
    remove_list = []
    i = 0
    while i < len(net_layers):
        if net.layer[i].type == 'Convolution':
            conv_param = net.layer[i].convolution_param
            conv_name = net.layer[i].name
            conv_layer = get_layer(weight, conv_name)
            i += 1
            if conv_param.group != 1:
                continue
            if i < len(net_layers) and net.layer[i].type == 'BatchNorm':
                batch_norm_param = net.layer[i].batch_norm_param
                batch_norm_name = net.layer[i].name
                batch_norm_layer = get_layer(weight, batch_norm_name)
                i += 1
                if net.layer[i].HasField('batch_norm_param') and not batch_norm_param.use_global_stats:
                    continue
                if i < len(net_layers) and net.layer[i].type == 'Scale':
                    scale_param = net.layer[i].scale_param
                    scale_name = net.layer[i].name
                    scale_layer = get_layer(weight, scale_name)
                    i += 1

                    print('fuse (%s, %s, %s)'%(conv_name, batch_norm_name, scale_name))
                    # weight, bias
                    conv_weight = convert_blob_to_array(conv_layer.blobs[0])
                    if conv_param.bias_term:
                        conv_bias = convert_blob_to_array(conv_layer.blobs[1])
                    else:
                        channels = conv_param.num_output
                        conv_bias = np.zeros(channels, dtype=np.float32)
                    # mean, std
                    mean = convert_blob_to_array(batch_norm_layer.blobs[0])
                    std = convert_blob_to_array(batch_norm_layer.blobs[1])
                    scale_factor = convert_blob_to_array(batch_norm_layer.blobs[2])
                    mean /= scale_factor
                    std /= scale_factor
                    # scale, shift
                    scale = convert_blob_to_array(scale_layer.blobs[0])
                    if scale_param.bias_term:
                        shift = convert_blob_to_array(scale_layer.blobs[1])
                    else:
                        shift = np.zeros(scale.shape, dtype=np.float32)
                    # eps
                    if batch_norm_param is not None:
                        eps = batch_norm_param.eps
                    else:
                        eps = 1e-5
                    # fuse
                    conv_weight, conv_bias = fuse(conv_weight, conv_bias, mean, std, scale, shift, eps)

                    conv_layer.blobs.pop()
                    if conv_param.bias_term:
                        conv_layer.blobs.pop()
                    else:
                        conv_param.bias_term = True
                    conv_layer.blobs.extend([convert_array_to_blob(conv_weight), convert_array_to_blob(conv_bias)])

                    # remove batchnorm and scale layer
                    weight.layer.remove(batch_norm_layer)
                    weight.layer.remove(scale_layer)
                    net.layer[i - 3].top[0] = net.layer[i - 1].top[0]
                    remove_list.extend([net.layer[i - 1], net.layer[i - 2]])
        else:
            i += 1

    for layer in remove_list:
        net.layer.remove(layer)
    # save
    out_net = '_nobn'.join(os.path.splitext(args.net))
    out_weight = '_nobn'.join(os.path.splitext(args.weight))
    with open(out_net, 'w') as fout:
        fout.write(text_format.MessageToString(net))
    with open(out_weight, 'wb') as fout:
        fout.write(weight.SerializeToString())


if __name__ == '__main__':
    main()
