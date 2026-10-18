# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for RISCV64 +D/F opmem operations
"""

from ...op.constraint import minmax_constraint
from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    opmem_modifier as mod
)

from ....registers import asm_data_type as adt


def make_riscv64_opmem_signatures():
    """
    Generate signatures for RISCV64 +D/F opmem operations
    """
    sigs = []

    floats = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
    ints = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
            adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

    for dt in floats:
        # normal fld without offset
        sigs.append(sig(
        modifiers=set(),
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, rt.FP, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ))
        # normal fld with immediate offset
        sigs.append(sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, rt.FP, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-2048, maxval=2047)])
            }
        ))

    # Same for integers
    for dt in ints:
        sigs.append(sig(
        modifiers=set(),
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, rt.GP, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ))
        sigs.append(sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, rt.GP, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-2048, maxval=2047)])
            }
        ))

    return sigs


def make_riscv64_prefetch_signatures():
    """
    Generate signatures for RISCV64 +D/F (Zicbop) prefetch operations
    """
    return [
        # prefetch without offset
        sig(
        modifiers=set(),
        structural_params={},
        operands={
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ),
        # prefetch with immediate offset
        sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-2048, maxval=2047)])
            }
        ),
    ]
