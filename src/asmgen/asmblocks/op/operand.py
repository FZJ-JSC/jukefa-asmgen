# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operands
"""

from dataclasses import dataclass,field
from enum import Enum, auto

from typing import Any


from ...registers import (
    asm_data_type as adt,
    greg_base,
    freg_base,
    vreg_base,
    treg_base,
    mreg_base
)

from .constraint import operand_constraint

class operand_type(Enum):
    """
    Valid types for operands, currently only registers and immediates
    """
    REGISTER = auto()
    IMMEDIATE = auto()

class operand_role(Enum):
    """
    Which role the operand plays in the operation
    """
    DATA = auto()
    ADDRESS = auto()
    MASK = auto()
    PARAM = auto()


class register_type(Enum):
    """
    Valid register types
    """
    GP = auto()
    FP = auto()
    VEC = auto()
    TILE = auto()
    MASK = auto()

class operand_modifier(Enum):
    """
    Modifiers on an operand
    """
    ILANE      = auto() # Selects a lane within a vector register with an immediate
    GLANE      = auto() # Selects a lane within a vector register with a gp reg
    BLOCKLANE  = auto() # Selects a lane within each block with `blocksize` elements
    BLOCK      = auto() # Operates on blocks of `blocksize` elements; with BCAST one
                        # block is replicated into every block (SVE ld1rq)
    IBLOCK     = auto() # Selects a block of `blocksize` elements with an immediate
    ILANESHIFT = auto()    # the source's lanes moved by irot lanes - negative towards
                           # lane 0, positive away; the lanes vacated hold what an
                           # *OTHERS modifier says
    ILANEROT   = auto()    # the source's lanes rotated by irot lanes
    IBLOCKSHIFT = auto()   # blocks of blocksize elements moved by irot blocks
    IBLOCKROT  = auto()    # blocks of blocksize elements rotated by irot blocks
    FILLOTHERS = auto()    # the lanes a shift vacates take the value in fillreg
    INVALIDOTHERS = auto() # the lanes a shift vacates hold anything (RVV's tail-
                           # agnostic slides)
    ZEROOTHERS = auto() # A lane load that doesn't preserve the lanes above the one it
                        # loads (AVX's vmovsd clears them; VEX vmovhpd and vinsertps
                        # clear all above the low 128 bits): load lanes in increasing
                        # order, and fill the upper blocks otherwise
    ROW        = auto() # Selects a row within a tile register
    COL        = auto() # Selects a column within a tile register
    BCAST      = auto() # Broadcasts a scalar across a vector/tile register
    VF         = auto() # Vector is replaced by a scalar
    ISHIFT = auto()     # the operand shifted left by an immediate first (bdreg_shift): add, sub
    VLSCALE = auto()    # an immediate counting vector lengths, in bytes: SVE's addvl, incb


def is_register_type(val : Any, rt : register_type) -> bool:
    """
    Tests if val is of the specified register type
    """
    if register_type.GP   == rt and isinstance(val, greg_base):
        return True
    if register_type.FP   == rt and isinstance(val, freg_base):
        return True
    if register_type.VEC  == rt and isinstance(val, vreg_base):
        return True
    if register_type.TILE == rt and isinstance(val, treg_base):
        return True
    if register_type.MASK == rt and isinstance(val, mreg_base):
        return True

    return False

@dataclass
class operand_shape:
    """
    Structural requirement for an operand
    """

    otype     : operand_type
    orole     : operand_role
    rtype     : register_type | None = None
    dt        : adt|None = None
    modifiers : set[operand_modifier] = field(default_factory=set)
    value_constraints: list[operand_constraint] = field(default_factory=list)


    def __post_init__(self):
        if not isinstance(self.otype, operand_type):
            raise TypeError(
                    f"otype must be operand_type, got {type(self.otype).__name__}")
        if not isinstance(self.orole, operand_role):
            raise TypeError(
                    f"orole must be operand_role, got {type(self.orole).__name__}")
        if self.orole in (operand_role.DATA, operand_role.MASK) \
                and self.otype != operand_type.REGISTER:
            raise ValueError(f"{self.orole.name} operand must be a register")
