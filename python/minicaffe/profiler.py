# coding = utf-8
# pylint: disable=invalid-name
"""Profiler in mini-caffe"""
from .base import LIB
from .base import c_str, check_call


class Profiler(object):
    """Profiler
    """

    @staticmethod
    def enable():
        """enable profiler
        """
        check_call(LIB.CaffeProfilerEnable())

    @staticmethod
    def disable():
        """disable profiler
        """
        check_call(LIB.CaffeProfilerDisable())

    @staticmethod
    def open_scope(name):
        """open a scope on profiler

        Parameters
        ----------
        name: string
            scope name
        """
        check_call(LIB.CaffeProfilerScopeStart(c_str(name)))

    @staticmethod
    def close_scope():
        """close a scope on profiler
        """
        check_call(LIB.CaffeProfilerScopeEnd())

    @staticmethod
    def dump(fn):
        """dump profiler data to fn

        Parameters
        ----------
        fn: string
            file path to save profiler data
        """
        check_call(LIB.CaffeProfilerDump(c_str(fn)))
