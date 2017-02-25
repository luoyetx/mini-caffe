# coding = utf-8
"""Net represents caffe::Net in C++"""
from __future__ import absolute_import
import ctypes
from .base import LIB
from .base import c_str, check_call
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

    def forward(self):
        """forward network, need to fill data blobs before call this function
        """
        check_call(LIB.CaffeNetForward(self.handle))
