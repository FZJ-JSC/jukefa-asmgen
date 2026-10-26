# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation/Instruction abstractions and utilities
"""


from .constraint import operand_constraint
from .modifier import operation_modifier
from .signature import operation_signature

from .operation import (
    operation,
)

from .operand import (
    operand_modifier,
    operand_shape,
    operand_type,
    operand_role,
    register_type
)

from .opmove import (
    opmove,
    opmove_modifier
)

from .opmem import (
    opmem_modifier,
    opmem_action,
    opmem
)

from .optransform import (
    widening_method,
    optransform_modifier,
    optransform,
    dummy_optransform)

from .misc import make_ord_prefix


#: the names before 0.6's phase 10 - deprecated for a release
_RENAMED = {"opd3": "optransform", "opd3_modifier": "optransform_modifier",
            "dummy_opd3": "dummy_optransform", "move_modifier": "opmove_modifier",
            "dummy_move": "dummy_opmove"}


def __getattr__(name):
    """
    opd3 is optransform, move's base class opmove (0.6): their old names,
    deprecated
    """
    if name in _RENAMED:
        import warnings
        warnings.warn(f"asmgen: asmgen.asmblocks.op.{name} is deprecated, use {_RENAMED[name]}",
                      DeprecationWarning, stacklevel=2)
        return globals()[_RENAMED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
