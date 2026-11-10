# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation/Instruction abstractions and utilities
"""


from .constraint import operand_constraint
from .misc import make_ord_prefix
from .modifier import operation_modifier
from .operand import (
    operand_modifier,
    operand_role,
    operand_shape,
    operand_type,
    register_type,
)
from .operation import (
    operation,
)
from .opmem import opmem, opmem_action, opmem_modifier
from .opmove import opmove, opmove_modifier
from .optransform import (
    dummy_optransform,
    optransform,
    optransform_modifier,
    widening_method,
)
from .signature import operation_signature
