# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for NEON opd3 operations
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

from ...op.constraint import minmax_constraint
from ...op.opd3 import widening_method as wm
from ....registers import asm_data_type as adt, adt_size
from ..types import neon_vreg

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_SIGNED_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
_UNSIGNED_INTS = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

_WIDENING_2X_MAP = {
    adt.FP8E4M3: adt.FP16,   adt.FP8E5M2: adt.FP16,
    adt.FP16: adt.FP32,
    adt.UINT8: adt.UINT16,   adt.SINT8: adt.SINT16,
    adt.UINT16: adt.UINT32,  adt.SINT16: adt.SINT32,
}
_WIDENING_4X_MAP = {
    adt.FP8E4M3: adt.FP32, adt.FP8E5M2: adt.FP32,
    adt.UINT8: adt.UINT32, adt.SINT8: adt.SINT32
}

_MIXED_INTS = [
    (adt.UINT8, adt.SINT8, adt.SINT16), (adt.SINT8, adt.UINT8, adt.SINT16),
    (adt.UINT16, adt.SINT16, adt.SINT32), (adt.SINT16, adt.UINT16, adt.SINT32),
]

# Readable enough, no need for subfunctions
# pylint: disable-next=too-many-branches
def make_neon_opd3_signatures(supports_np: bool, widens: bool = True) -> list[sig]:
    """
    Generate signatures for NEON opd3 operations
    """
    sigs = []

    base_mods = [set()]
    if supports_np:
        base_mods.extend([{mod.NP}])

    # Leave out c for now (Need to read up how exactly FDOTA works with C lanes)
    opd_mods_list = [{}, {'bdreg':{opd_mod.ILANE}}]

    def add_sig (*, dts, mods, opd_mods, is_widening=False):
        types = set(dts.values())
        # FP8 arithmetic is widening FMAs only; floating-point widening is FMAs
        # only (fmlal...): there is no faddl or fmull for floats
        if types & {adt.FP8E4M3, adt.FP8E5M2} and not is_widening:
            return
        if is_widening and not widens and types & {adt.FP16, adt.BF16} | {adt.FP8E4M3, adt.FP8E5M2}:
            return
        # FP16 to FP32 by halves (fmlal/fmlal2); BF16 and FP8 by bottom and top
        method = wm.SPLIT_CONTIGUOUS if (adt.FP16, adt.FP32) == (dts['adreg'], dts['cdreg']) \
            else wm.SPLIT_INTERLEAVED
        struct_params = {'widening_method': method} if is_widening else {}

        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dts['adreg']),
            'bdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dts['bdreg']),
            'cdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dts['cdreg'])
        }

        for opd, omods in opd_mods.items():
            ops[opd].modifiers = omods
            if {opd_mod.ILANE,opd_mod.BLOCKLANE}.intersection(omods):
                max_lane = (16 // adt_size(dts[opd]))-1
                ops[f"{opd}_lane"] = osh(
                        ot.IMMEDIATE, orl.PARAM, None, None,
                        value_constraints=[
                            minmax_constraint(minval=0,maxval=max_lane)]
                        )
                if opd_mod.BLOCKLANE in omods:
                    struct_params[f"{opd}_blocksize"] = max_lane+1

                # with 16bit indexed fma, b has to be v0-v15
                if 'bdreg' == opd and adt_size(dts[opd]) <= 2:
                    ops[opd].value_constraints.append(
                            minmax_constraint(
                                what='index',
                                getint=lambda reg : reg.idx,
                                makeval=lambda idx : neon_vreg(reg_idx=idx),
                                minval=0, maxval=15
                            )
                    )

        if mod.PART in mods:
            max_part = (adt_size(dts['cdreg']) // adt_size(dts['adreg'])) - 1
            ops['part'] = osh(
                ot.IMMEDIATE, orl.PARAM, None, None,
                value_constraints=[minmax_constraint(minval=0, maxval=max_part)]
            )

        requires = set()
        if (adt.FP16, adt.FP32) == (dts['adreg'], dts['cdreg']):
            requires.add("FHM")
        elif adt.FP16 in types and not is_widening:
            requires.add("FP16")
        if adt.BF16 in types:
            requires.add("BF16")
        if types & {adt.FP8E4M3, adt.FP8E5M2}:
            requires.add("FP8FMA")
        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            requires=frozenset(requires),
            operands=ops
        ))

    def make_dt_dict(dt: adt, widening_map : dict|None = None) -> dict[str,adt]:
        dts = {
            'adreg' : dt,
            'bdreg' : dt,
        }
        if widening_map is not None:
            dts['cdreg'] = widening_map[dt]
        else:
            dts['cdreg'] = dt

        return dts


    for dt in _FLOATS:
        for m in base_mods:
            for om in opd_mods_list:
                add_sig(dts=make_dt_dict(dt), mods=m, opd_mods=om)
                if dt in _WIDENING_2X_MAP:
                    add_sig(dts=make_dt_dict(dt, _WIDENING_2X_MAP),
                            mods=m | {mod.PART}, opd_mods=om,
                            is_widening=True)
                if dt in _WIDENING_4X_MAP:
                    add_sig(dts=make_dt_dict(dt, _WIDENING_4X_MAP),
                            mods=m | {mod.PART}, opd_mods=om,
                            is_widening=True)

    for dt in _SIGNED_INTS:
        for m in base_mods:
            for om in opd_mods_list:
                add_sig(dts=make_dt_dict(dt), mods=m, opd_mods=om)
                if dt in _WIDENING_2X_MAP:
                    add_sig(dts=make_dt_dict(dt, _WIDENING_2X_MAP),
                            mods=m | {mod.PART}, opd_mods=om,
                            is_widening=True)
                if dt in _WIDENING_4X_MAP:
                    add_sig(dts=make_dt_dict(dt, _WIDENING_4X_MAP),
                            mods=m | {mod.PART}, opd_mods=om,
                            is_widening=True)

    for dt in _UNSIGNED_INTS:
        for m in base_mods:
            for om in opd_mods_list:
                # Widening only for unsigned ints
                if dt in _WIDENING_2X_MAP:
                    add_sig(dts=make_dt_dict(dt, _WIDENING_2X_MAP),
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)
                if dt in _WIDENING_4X_MAP:
                    add_sig(dts=make_dt_dict(dt, _WIDENING_4X_MAP),
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)

    for a_dt, b_dt, c_dt in _MIXED_INTS:
        for om in opd_mods_list:
            add_sig(dts={'adreg':a_dt,'bdreg':b_dt,'cdreg':c_dt},
                    mods={mod.PART},
                    opd_mods=om,
                    is_widening=True)

    return sigs
