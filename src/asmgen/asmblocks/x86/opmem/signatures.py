# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for Base X86_64 opmem operations
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

from ....registers import asm_data_type as adt, adt_is_float

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

def make_x86_64_opmem_signatures():
    """
    Generate signatures for Base X86_64 opmem operations
    """
    sigs = []


    for dt in _FLOATS+_INTS:

        a_rt = rt.FP if adt_is_float(dt) else rt.GP

        # (f)ld without offset
        sigs.append(sig(
        modifiers=set(),
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, a_rt, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ))
        # (f)ld with immediate offset
        sigs.append(sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, a_rt, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-2**31, maxval=2**31-1)])
            }
        ))
        # (f)ld with gp-reg offset
        sigs.append(sig(
        modifiers={mod.GOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, a_rt, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'offreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ))

        # (f)ld with gp-reg and immediate offset
        sigs.append(sig(
        modifiers={mod.GOFFSET, mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, a_rt, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'offreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-2**31, maxval=2**31-1)])
            }
        ))

    return sigs


def make_x86_prefetch_signatures():
    """
    Generate signatures for Base X86_64 prefetch operations
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
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-2**31, maxval=2**31-1)])
            }
        ),
        # prefetch with gp-reg offset
        sig(
        modifiers={mod.GOFFSET},
        structural_params={},
        operands={
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'offreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ),
    ]
