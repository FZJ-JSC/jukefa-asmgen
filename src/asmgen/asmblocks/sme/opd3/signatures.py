# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for SME opd3 operations
"""
from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    opd3_modifier as mod
)

from ...op.opd3 import widening_method as wm
from ....registers import asm_data_type as adt, adt_size

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
_SIGNED_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
_UNSIGNED_INTS = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

_WIDENING_2X_MAP = {
    adt.FP8E4M3: adt.FP16,   adt.FP8E5M2: adt.FP16,
    adt.FP16: adt.FP32,
    adt.BF16: adt.FP32,
    adt.UINT16: adt.UINT32,  adt.SINT16: adt.SINT32,
}
_WIDENING_4X_MAP = {
    adt.FP8E4M3: adt.FP32, adt.FP8E5M2: adt.FP32,
    adt.UINT8: adt.UINT32, adt.SINT8: adt.SINT32
}

_MIXED_INTS = [
    (adt.UINT8, adt.SINT8, adt.SINT32), (adt.SINT8, adt.UINT8, adt.SINT32),
    (adt.UINT16, adt.SINT16, adt.SINT64), (adt.SINT16, adt.UINT16, adt.SINT64),
]

# Readable enough, no need for subfunctions
# pylint: disable-next=too-many-branches
def make_sme_opd3_signatures(supports_np: bool) -> list[sig]:
    """
    Generate signatures for SME opd3 operations
    """
    sigs = []

    base_mods = [{mod.MASK}]
    if supports_np:
        base_mods.extend([{mod.MASK,mod.NP}])


    def add_sig(a_dt, b_dt, c_dt, *, mods, is_widening=False):
        struct_params = {'widening_method': wm.DOT_NEIGHBOURS} if is_widening else {}

        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, a_dt),
            'bdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, b_dt),
            'cdreg': osh(ot.REGISTER, orl.DATA, rt.TILE, c_dt)
        }

        if mod.MASK in mods:
            ops['amreg'] = osh(ot.REGISTER, orl.MASK, rt.MASK, a_dt)
            ops['bmreg'] = osh(ot.REGISTER, orl.MASK, rt.MASK, b_dt)

        # FP8 only widens: there is no FP8 into FP8 outer product
        if a_dt in (adt.FP8E4M3, adt.FP8E5M2) and a_dt == c_dt:
            return
        # outer products beyond SME's own: FEAT_SME_F64F64, _F16F16, _B16B16,
        # _F8F16, _F8F32, _I16I64; the two-way integer ones SME2
        requires = set()
        if adt.FP64 == a_dt:
            requires.add("F64F64")
        elif (adt.FP16, adt.FP16) == (a_dt, c_dt):
            requires.add("F16F16")
        elif (adt.BF16, adt.BF16) == (a_dt, c_dt):
            requires.add("SME_B16B16")
        elif a_dt in (adt.FP8E4M3, adt.FP8E5M2):
            requires.add("F8F16" if adt.FP16 == c_dt else "F8F32")
        elif 2 == adt_size(a_dt) and 8 == adt_size(c_dt):
            requires.add("I16I64")
        elif 2 == adt_size(a_dt) and 4 == adt_size(c_dt) and a_dt not in (adt.FP16, adt.BF16):
            requires.add("SME2")
        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops,
            requires=frozenset(requires)
        ))

    for dt in _FLOATS:
        for m in base_mods:
            add_sig(dt, dt, dt, mods=m)
            if dt in _WIDENING_2X_MAP:
                add_sig(dt, dt, _WIDENING_2X_MAP[dt], mods=m, is_widening=True)
            if dt in _WIDENING_4X_MAP:
                add_sig(dt, dt, _WIDENING_4X_MAP[dt], mods=m, is_widening=True)

    for dt in _SIGNED_INTS:
        for m in base_mods:
            if dt in _WIDENING_2X_MAP:
                add_sig(dt, dt, _WIDENING_2X_MAP[dt], mods=m, is_widening=True)
            if dt in _WIDENING_4X_MAP:
                add_sig(dt, dt, _WIDENING_4X_MAP[dt], mods=m, is_widening=True)

    for dt in _UNSIGNED_INTS:
        for m in base_mods:
            if dt in _WIDENING_2X_MAP:
                add_sig(dt, dt, _WIDENING_2X_MAP[dt], mods=m, is_widening=True)
            if dt in _WIDENING_4X_MAP:
                add_sig(dt, dt, _WIDENING_4X_MAP[dt], mods=m, is_widening=True)

    for a_dt,b_dt,c_dt in _MIXED_INTS:
        for m in base_mods:
            add_sig(a_dt, b_dt, c_dt, mods=m, is_widening=True)

    return sigs
