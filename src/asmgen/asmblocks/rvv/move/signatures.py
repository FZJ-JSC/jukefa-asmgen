# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Signatures for RVV move instructions: copies, broadcasts and moves into
element 0
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
from ....registers import asm_data_type as adt

_FLOATS = [adt.FP64, adt.FP32, adt.FP16]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]


def make_rvv_move_signatures() -> list[opsig]:
    """
    adreg is the source, bdreg the destination; the element width comes from
    vsetvli:

        copy              vmv.v.v     vB, vA
        scalar to all     vfmv.v.f    vB, fA     (vmv.v.x for integers)
        lane to all       vrgather.vi vB, vA, i  adreg ILANE, bdreg BCAST;
                                                 i within 0..31, vB not vA
        scalar to lane 0  vfmv.s.f    vB, fA     (vmv.s.x), bdreg ILANE
    """
    sigs = []

    def add_sig(dt, amods, bmods):
        scalar = rgt.FP if dt in _FLOATS else rgt.GP
        operands = {
            'adreg': osh(otype=ot.REGISTER, orole=orl.DATA,
                         rtype=rgt.VEC if (omod.ILANE in amods or not (amods | bmods))
                         else scalar, dt=dt),
            'bdreg': osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
        }
        if omod.ILANE in amods:
            operands['adreg_lane'] = osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
                minmax_constraint(minval=0, maxval=31)])
        if omod.ILANE in bmods:
            operands['bdreg_lane'] = osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
                minmax_constraint(minval=0, maxval=0)])
        operands['adreg'].modifiers = amods
        operands['bdreg'].modifiers = bmods
        sigs.append(opsig(modifiers=set(), structural_params={}, operands=operands))

    for dt in _FLOATS + _INTS:
        add_sig(dt, set(), set())
        add_sig(dt, set(), {omod.BCAST})
        add_sig(dt, {omod.ILANE}, {omod.BCAST})
        add_sig(dt, set(), {omod.ILANE})
    return sigs
