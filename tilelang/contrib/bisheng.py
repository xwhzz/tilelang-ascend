# pylint: disable=invalid-name
# modified from apache tvm python/tvm/contrib/nvcc.py
"""Utility to invoke nvcc compiler in the system"""
from __future__ import annotations

import os
import subprocess
import warnings
import contextlib
from tilelang.env import ASCEND_HOME
import shutil
import tempfile
import tvm_ffi
from tilelang import tvm as tvm
from tvm.target import Target

from tvm.base import py_str
from tvm.contrib import utils



def find_ascend_path():
    """Utility function to find ascend path

    Returns
    -------
    path : str
        Path to cuda root.
    """
    if ASCEND_HOME:
        return ASCEND_HOME
    raise RuntimeError(
        "Failed to automatically detect Ascend installation."
    )

def is_a5(target):
    return False
