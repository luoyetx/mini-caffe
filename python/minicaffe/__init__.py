# coding = utf-8
"""Mini-Caffe: A minimal runtime core of Caffe, Forward only and GPU support"""
from .net import Net
from .base import check_gpu_available, set_runtime_mode
from .craft import LayerCrafter
from .profiler import Profiler

__version__ = '0.4.0'
