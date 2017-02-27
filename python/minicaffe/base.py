# coding = utf-8
# pylint: disable=invalid-name, no-member
"""lib informantion about mini-caffe"""
from __future__  import absolute_import
import os
import sys
import ctypes
import numpy as np


# types
BlobHandle = ctypes.c_void_p
NetHandle = ctypes.c_void_p
real_t = ctypes.c_float


def find_lib_path():
    """find libcaffe.so or caffe.dll

    Returns
    -------
    lib_path: list(string)
        List of all found path to load caffe library
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dll_path = [current_dir,
                os.path.join(current_dir, '../../build'),
                os.path.join(current_dir, '../../build/Release')]
    if os.name == 'nt':
        dll_path = [os.path.join(p, 'caffe.dll') for p in dll_path]
    else:
        dll_path = [os.path.join(p, 'libcaffe.so') for p in dll_path]
    lib_path = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if len(lib_path) == 0:
        raise RuntimeError('Cannot find caffe library.\n' +
                           'List of candidates:\n' + str('\n'.join(dll_path)))
    return lib_path


def load_lib():
    """load caffe library"""
    lib_path = find_lib_path()
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    # change default int ret
    lib.CaffeGetLastError.restype = ctypes.c_char_p
    lib.CaffeBlobData.restype = ctypes.POINTER(real_t)
    return lib


LIB = load_lib()

if sys.version_info[0] == 3:
    string_type = str
    py_str = lambda x: x.decode('utf-8')
    c_str = lambda x: ctypes.c_char_p(x.encode('utf-8'))
else:
    string_type = basestring
    py_str = lambda x: x
    c_str = lambda x: ctypes.c_char_p(x)


class CaffeError(Exception):
    """Error that will be throwed by all caffe functions"""
    pass


def check_call(ret):
    """check the return value of C API call
    This function will raise  exception when error occurs.
    Wrap all C API call with this function

    Parameters
    ----------
    ret: int
        return value of C API call
    """
    if ret != 0:
        raise CaffeError(py_str(LIB.CaffeGetLastError()))


def ctypes2numpy_shared(cptr, shape):
    """convert a ctypes pointer to a numpy array, share the same memory

    Parameters
    ----------
    cptr: ctypes.POINTER(real_t)
        pointer to the memory
    shape: tuple(int)
        shape of the array

    Returns
    -------
    out: numpy array
        the array
    """
    if not isinstance(cptr, ctypes.POINTER(real_t)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (real_t * size).from_address(ctypes.addressof(cptr.contents))
    return np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)


def check_gpu_available():
    """check gpu available

    Returns
    -------
    aval: bool
        True if we can use GPU
    """
    ret = LIB.CaffeGPUAvailable()
    ret = True if ret == 1 else False
    return ret


def set_runtime_mode(mode, device_id=-1):
    """set mini-caffe runtime mode, CPU or GPU with device
    If device is not available, an Error will be throwed

    Parameters
    ----------
    mode: int
        0 for CPU and 1 for GPU
    device_id: int
        device id for GPU, -1 for CPU
    """
    assert mode == 0 or mode == 1
    check_call(LIB.CaffeSetMode(mode, device_id))
