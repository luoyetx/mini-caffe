# coding = utf-8
"""Blob represents caffe::Blob"""
from __future__ import absolute_import
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
        s: list(int)
            num, channels, height, width
        """
        num = LIB.CaffeBlobNum(self.handle)
        channels = LIB.CaffeBlobChannels(self.handle)
        height = LIB.CaffeBlobHeight(self.handle)
        width = LIB.CaffeBlobWidth(self.handle)
        return [num, channels, height, width]

    def reshape(self, num, channels=1, height=1, width=1):
        """reshape this blob, this also affect the internal data buffer.
        Data return by `data` may be invalid

        Parameters
        ----------
        num, channels, height, width: int
            blob num, channels, height, width
        """
        check_call(LIB.CaffeBlobReshape(self.handle, num, channels, height, width))

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
