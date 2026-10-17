# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Signatures for NEON move instructions: copies, broadcasts (dup) and lane
inserts (ins)
"""

from ...op import (
    operation_signature as opsig,
    operand_modifier as omod,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rgt,
)
from ...op.constraint import minmax_constraint
from ....registers import asm_data_type as adt, adt_size

_FLOATS = [adt.FP64, adt.FP32, adt.FP16]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]


def make_neon_move_signatures() -> list[opsig]:
    """
    adreg is the source, bdreg the destination:

        copy             mov  vB.16b, vA.16b
        scalar to all    dup  vB.2d, vA.d[0]      (dup vB.4s, wA for integers)
        lane to all      dup  vB.2d, vA.d[i]      adreg ILANE, bdreg BCAST
        scalar to lane   ins  vB.d[j], vA.d[0]    bdreg ILANE
        lane to lane     ins  vB.d[j], vA.d[i]    adreg ILANE, bdreg ILANE

    A scalar FP register is lane 0 of the vector register with its number.
    """
    sigs = []

    def lane(dt):
        return osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
            minmax_constraint(minval=0, maxval=16 // adt_size(dt) - 1)])

    def add_sig(dt, amods, bmods):
        scalar = rgt.FP if dt in _FLOATS else rgt.GP
        operands = {
            'adreg': osh(otype=ot.REGISTER, orole=orl.DATA,
                         rtype=rgt.VEC if (omod.ILANE in amods or not (amods | bmods))
                         else scalar, dt=dt),
            'bdreg': osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
        }
        if omod.ILANE in amods:
            operands['adreg_lane'] = lane(dt)
        if omod.ILANE in bmods:
            operands['bdreg_lane'] = lane(dt)
        operands['adreg'].modifiers = amods
        operands['bdreg'].modifiers = bmods
        sigs.append(opsig(modifiers=set(), structural_params={}, operands=operands))

    for dt in _FLOATS + _INTS:
        add_sig(dt, set(), set())
        add_sig(dt, set(), {omod.BCAST})
        add_sig(dt, {omod.ILANE}, {omod.BCAST})
        add_sig(dt, set(), {omod.ILANE})
        add_sig(dt, {omod.ILANE}, {omod.ILANE})
    return sigs
