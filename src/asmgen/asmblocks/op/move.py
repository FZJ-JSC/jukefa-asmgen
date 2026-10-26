# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
move is opmove since 0.6 (phase 10): this module and the names in it are
deprecated for a release - asmgen.asmblocks.op.opmove
"""
import warnings as _warnings

import importlib as _importlib

# the module: the package's name opmove is its class
_opmove = _importlib.import_module(".opmove", __package__)

_RENAMED = {'move': 'opmove', 'move_modifier': 'opmove_modifier', 'dummy_move': 'dummy_opmove'}


def __getattr__(name):
    new = _RENAMED.get(name, name)
    if name.startswith("__") or not hasattr(_opmove, new):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _warnings.warn(f"asmgen: asmgen.asmblocks.op.move.{name} is deprecated, "
                   f"use asmgen.asmblocks.op.opmove.{new}", DeprecationWarning, stacklevel=2)
    return getattr(_opmove, new)
