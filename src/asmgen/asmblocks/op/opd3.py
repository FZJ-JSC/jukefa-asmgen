# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
opd3 is optransform since 0.6 (phase 10): this module and the names in it are
deprecated for a release - asmgen.asmblocks.op.optransform
"""
import warnings as _warnings

import importlib as _importlib

# the module: the package's name optransform is its class
_optransform = _importlib.import_module(".optransform", __package__)

_RENAMED = {'opd3': 'optransform', 'opd3_modifier': 'optransform_modifier', 'dummy_opd3': 'dummy_optransform'}


def __getattr__(name):
    new = _RENAMED.get(name, name)
    if name.startswith("__") or not hasattr(_optransform, new):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _warnings.warn(f"asmgen: asmgen.asmblocks.op.opd3.{name} is deprecated, "
                   f"use asmgen.asmblocks.op.optransform.{new}", DeprecationWarning, stacklevel=2)
    return getattr(_optransform, new)
