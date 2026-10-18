# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for AArch64  opmem operations
"""

from ....registers import adt_size
from ...op.constraint import either_constraint, minmax_constraint, multiple_constraint
from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    opmem_modifier as mod
)

from ....registers import asm_data_type as adt, adt_is_float

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]


def _scale(dt) -> int:
    """
    What an unsigned immediate offset is scaled by: the access's size - a
    prefetch's (no data) 8, as prfm's
    """
    return 8 if dt is None else adt_size(dt)

def make_aarch64_opmem_signatures():
    """
    Generate signatures for AArch64 opmem operations
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
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[either_constraint(options=[
                [minmax_constraint(minval=0, maxval=4095*_scale(dt)), multiple_constraint(multiple=_scale(dt))],
                [minmax_constraint(minval=-256, maxval=255)]])])
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
        # (f)ld with postinc immediate offset
        sigs.append(sig(
        modifiers={mod.POSTINC},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, a_rt, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'iinc': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[minmax_constraint(minval=-256, maxval=255)])
            }
        ))
        # (f)ld with postinc gp-reg offset
        sigs.append(sig(
        modifiers={mod.POSTINC},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, orl.DATA, a_rt, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'increg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            }
        ))

    return sigs


def make_aarch64_prefetch_signatures():
    """
    Generate signatures for AArch64 prefetch operations
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
            'ioffset': osh(ot.IMMEDIATE, orl.PARAM, None, adt.SINT64, value_constraints=[either_constraint(options=[
                [minmax_constraint(minval=0, maxval=4095*_scale(None)), multiple_constraint(multiple=_scale(None))],
                [minmax_constraint(minval=-256, maxval=255)]])])
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
