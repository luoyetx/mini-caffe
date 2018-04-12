# coding = utf-8
# pylint: disable=invalid-name
"""Net represents caffe::Net in C++"""
from __future__ import absolute_import
from collections import defaultdict
import ctypes
from .base import LIB
from .base import c_str, py_str, check_call
from .base import NetHandle, BlobHandle
from .blob import Blob


class Net(object):
    """Net in caffe
    """

    def __init__(self, prototxt, caffemodel):
        """create an empty net object

        Parameters
        ----------
        prototxt: string
            caffe network prototxt file path
        caffemodel: string
            caffe network caffemodel file path
        """
        self.handle = NetHandle()
        check_call(LIB.CaffeNetCreate(c_str(prototxt),
                                      c_str(caffemodel),
                                      ctypes.byref(self.handle)))

    def __del__(self):
        """destruct object
        """
        check_call(LIB.CaffeNetDestroy(self.handle))

    def get_blob(self, name):
        """get blob by name

        Parameters
        ----------
        name: string
            blob name in network buffer

        Returns
        -------
        blob: Blob
            network internal buffer
        """
        handle = BlobHandle()
        check_call(LIB.CaffeNetGetBlob(self.handle, c_str(name), ctypes.byref(handle)))
        return Blob(handle)

    @property
    def blobs(self):
        """return network internal blobs

        Returns
        -------
        blobs: dict(name: Blob)
            network internal blobs with their name
        """
        ctypes_n = ctypes.c_int32()
        ctypes_names = ctypes.POINTER(ctypes.c_char_p)()
        ctypes_blobs = ctypes.POINTER(BlobHandle)()
        check_call(LIB.CaffeNetListBlob(self.handle, ctypes.byref(ctypes_n),
                                        ctypes.byref(ctypes_names),
                                        ctypes.byref(ctypes_blobs)))
        blobs = dict()
        for i in range(ctypes_n.value):
            name = py_str(ctypes_names[i])
            blob = Blob(BlobHandle(ctypes_blobs[i]))
            blobs[name] = blob
        return blobs

    @property
    def params(self):
        """return network params

        Returns
        -------
        params: dict: layer_name -> list((param_name, Blob))
            list parameter by layers, every layer parameters listed in order with their name
        """
        ctypes_n = ctypes.c_int32()
        ctypes_names = ctypes.POINTER(ctypes.c_char_p)()
        ctypes_params = ctypes.POINTER(BlobHandle)()
        check_call(LIB.CaffeNetListParam(self.handle, ctypes.byref(ctypes_n),
                                         ctypes.byref(ctypes_names),
                                         ctypes.byref(ctypes_params)))
        params = defaultdict(list)
        for i in range(ctypes_n.value):
            name = py_str(ctypes_names[i])
            param = Blob(BlobHandle(ctypes_params[i]))
            layer_name = '_'.join(name.split('_')[:-1])
            params[layer_name].append((name, param))
        return params

    def mark_output(self, name):
        """mark network internal blob as output, you need to mark it if you need the data,
        and the blob is not an output blob, otherwise you may get the wrong result

        Parameters
        ----------
        name: string
            blob name to mark as an output
        """
        check_call(LIB.CaffeNetMarkOutput(self.handle, c_str(name)))

    def forward(self, **kwargs):
        """forward network, need to fill data blobs before call this function

        Parameters
        ==========
        kwargs: dict(str: np.array)
            input blob map
        """
        for k, v in kwargs.items():
            blob = self.get_blob(k)
            blob.reshape(*v.shape)
            blob.data[...] = v
        check_call(LIB.CaffeNetForward(self.handle))
