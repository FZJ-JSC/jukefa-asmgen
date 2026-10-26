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
