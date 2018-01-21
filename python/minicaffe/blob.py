# coding = utf-8
"""Blob represents caffe::Blob"""
from __future__ import absolute_import
import ctypes
from .base import LIB
from .base import check_call, ctypes2numpy_shared


class Blob(object):
    """Blob in caffe, users shouldn't create object through this class.
    Always gets a reference from Net object
    """

    def __init__(self, handle):
        """constructor from a ctype handle

        Parameters
        ----------
        handle: BlobHandle
            blob handle
        """
        self.handle = handle

    @property
    def shape(self):
        """get blob shape

        Returns:
        shape: list(int)
            shape of this blob
        """
        ctypes_n = ctypes.c_int32()
        ctypes_shape = ctypes.POINTER(ctypes.c_int32)()
        check_call(LIB.CaffeBlobShape(self.handle, ctypes.byref(ctypes_n),
                                      ctypes.byref(ctypes_shape)))
        shape = [ctypes_shape[i] for i in range(ctypes_n.value)]
        return shape

    def reshape(self, *shape):
        """reshape this blob, this also affect the internal data buffer.
        Data return by `data` may be invalid

        Parameters
        ----------
        shape: list(int)
            shape of this blob
        """
        shape_size = len(shape)
        ctypes_shape = (ctypes.c_int32 * shape_size)(*shape)
        check_call(LIB.CaffeBlobReshape(self.handle, shape_size, ctypes_shape))

    @property
    def data(self):
        """wrap internal data buffer with numpy array
        You can directly modify the data, but never change shape or size of the array.
        If you want to change the total size or shape of this blob, call `reshape` and
        then get data from this function again.

        Returns
        -------
        array: numpy.array
            array that wraps internal data buffer
        """
        shape = self.shape
        cptr = LIB.CaffeBlobData(self.handle)
        return ctypes2numpy_shared(cptr, shape)
