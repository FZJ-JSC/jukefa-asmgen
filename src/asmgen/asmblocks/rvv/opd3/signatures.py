# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for RVV opd3 operations
"""

from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    opd3_modifier as mod,
    operand_modifier as opd_mod
)

from ...op.opd3 import widening_method as wm
from ...op.constraint import multiple_constraint

from ....registers import asm_data_type as adt
from ..types import rvv_vreg

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

_WIDENING_MAP = {
    adt.FP8E4M3: adt.FP16,   adt.FP8E5M2: adt.FP16,
    adt.FP16: adt.FP32,      adt.FP32: adt.FP64,
    adt.UINT8: adt.UINT16,   adt.SINT8: adt.SINT16,
    adt.UINT16: adt.UINT32,  adt.SINT16: adt.SINT32,
    adt.UINT32: adt.UINT64,  adt.SINT32: adt.SINT64,
}

_MIXED_INTS = [
    (adt.UINT8, adt.SINT8, adt.SINT16), (adt.SINT8, adt.UINT8, adt.SINT16),
    (adt.UINT16, adt.SINT16, adt.SINT32), (adt.SINT16, adt.UINT16, adt.SINT32),
    (adt.UINT32, adt.SINT32, adt.SINT64), (adt.SINT32, adt.UINT32, adt.SINT64)
]

_RVV_IDX_2X = multiple_constraint(
    what="index",
    getint=lambda reg: reg.idx,
    makeval=lambda idx: rvv_vreg(reg_idx=idx),
    multiple=2
)

def make_rvv_opd3_signatures(supports_np: bool) -> list[sig]:
    """
    Generate signatures for RVV opd3 operations

    :param supports_np: Whether negate-product is supported (only FMA has it)
    """
    sigs = []

    np_options = [set(), {mod.NP}] if supports_np else [set()]

    def add_sig(a_dt, b_dt, c_dt, *, b_rt, mods, opd_mods=None, is_widening=False):
        if opd_mods is None:
            opd_mods = dict()
        struct_params = {'widening_method': wm.VEC_GROUP} if is_widening else {}
        constraints = [_RVV_IDX_2X] if is_widening else []

        b_mods = opd_mods.get('bdreg',set())
        requires = {"ZVFH"} if adt.FP16 in (a_dt, b_dt, c_dt) else set()
        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            requires=frozenset(requires),
            operands={
                'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, a_dt),
                'bdreg': osh(ot.REGISTER, orl.DATA, b_rt, b_dt, modifiers=b_mods),
                'cdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, c_dt,
                             value_constraints=constraints)
            }
        ))

    for dt in _FLOATS:
        for b_mods, b_rt in [(set(), rt.VEC), ({opd_mod.VF}, rt.FP)]:
            for np_mod in np_options:
                mods = np_mod

                add_sig(dt, dt, dt, b_rt=b_rt, mods=mods, opd_mods={'bdreg':b_mods})
                if dt in _WIDENING_MAP:
                    add_sig(dt, dt, _WIDENING_MAP[dt], b_rt=b_rt,
                            mods=mods,
                            opd_mods={'bdreg':b_mods},
                            is_widening=True)

    for dt in _INTS:
        for b_mods, b_rt in [(set(), rt.VEC), ({opd_mod.VF}, rt.GP)]:
            for np_mod in np_options:
                mods = np_mod

                add_sig(dt, dt, dt, b_rt=b_rt, mods=mods, opd_mods={'bdreg':b_mods})

                if dt in _WIDENING_MAP:
                    if mod.NP in mods:
                        continue
                    add_sig(dt, dt, _WIDENING_MAP[dt], b_rt=b_rt,
                            mods=mods,
                            opd_mods={'bdreg':b_mods},
                            is_widening=True)

    for a_dt, b_dt, c_dt in _MIXED_INTS:
        add_sig(a_dt, b_dt, c_dt, b_rt=rt.GP, mods=set(), is_widening=True)
        add_sig(a_dt, b_dt, c_dt, b_rt=rt.GP, mods=set(),
                opd_mods={'bdreg':{opd_mod.VF}}, is_widening=True)

    return sigs
