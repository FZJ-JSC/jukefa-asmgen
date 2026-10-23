# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Meta-module for NEON opd3 operations
"""
from .fma import *
from .mul import *
from .add import *


_RENAMED = {"neon_fadd": "neon_add", "neon_fmul": "neon_mul"}


def __getattr__(name):
    """
    The classes' names before 0.6 (neon_fadd, neon_fmul): deprecated
    """
    if name in _RENAMED:
        import warnings
        warnings.warn(f"asmgen: {name} is deprecated, use {_RENAMED[name]}",
                      DeprecationWarning, stacklevel=2)
        return globals()[_RENAMED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

