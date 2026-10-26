# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for SVE opd3 operations
"""
from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    optransform_modifier as mod,
    operand_modifier as opd_mod
)

from ...op.constraint import otherplusn_constraint, minmax_constraint
from ...op.optransform import widening_method as wm
from ....registers import asm_data_type as adt, adt_size
from ..types import sve_vreg

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
_SIGNED_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
_UNSIGNED_INTS = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

_WIDENING_2X_MAP = {
    adt.FP8E4M3: adt.FP16,   adt.FP8E5M2: adt.FP16,
    adt.FP16: adt.FP32,
    adt.BF16: adt.FP32,
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
def make_sve_opd3_signatures(supports_np: bool, destructive: bool = False,
                              unpredicated: bool = False, widens: bool = True) -> list[sig]:
    """
    Generate signatures for NEON opd3 operations
    """
    sigs = []

    base_mods = [set()]
    if supports_np:
        base_mods.extend([{mod.NP}])

    base_opd_mods = [{},{'bdreg' : {opd_mod.BLOCKLANE}}]


    def add_sig(a_dt, b_dt, c_dt, *, mods, opd_mods, is_widening=False):
        types = {a_dt, b_dt, c_dt}
        # FP8 arithmetic is widening FMAs only; floating-point widening is FMAs
        # only (fmlalb...): integers widen in adds and multiplies too (SVE2)
        if types & {adt.FP8E4M3, adt.FP8E5M2} and not is_widening:
            return
        if is_widening and not widens and types & {adt.FP16, adt.BF16} | {adt.FP8E4M3, adt.FP8E5M2}:
            return
        # bottom and top: even and odd elements
        struct_params = {'widening_method': wm.SPLIT_INTERLEAVED} if is_widening else {}

        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, a_dt),
            'bdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, b_dt),
            'cdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, c_dt,
                         value_constraints=[otherplusn_constraint(
                             other='adreg', offset=0, getint=lambda r: r.idx,
                             what="register")] if destructive and mod.MASK in mods else [])
        }

        if mod.MASK in mods:
            ops['amreg'] = osh(ot.REGISTER, orl.MASK, rt.MASK, c_dt)

        bdreg_mods = opd_mods.get('bdreg',set())

        ops['bdreg'].modifiers = bdreg_mods

        if opd_mod.BLOCKLANE in bdreg_mods:
            # SVE works on 128 bit chunks and the lane is selected in operand b
            blocksize = 16//adt_size(b_dt)
            struct_params['bdreg_blocksize'] = blocksize
            ops['bdreg_lane'] = osh(
                ot.IMMEDIATE, orl.PARAM, None, None,
                value_constraints=[minmax_constraint(minval=0, maxval=blocksize-1)]
            )
            # from ddi0602:
            # <Zm>
            # For the "Half-precision" and "Single-precision" variants: is the name
            # of the second source scalable vector register Z0-Z7, encoded in the "Zm" field.
	        # For the "Double-precision" variant: is the name of the second source
            # scalable vector register Z0-Z15, encoded in the "Zm" field.
            max_reg_idx = 15
            if adt_size(b_dt) <= 4:
                max_reg_idx = 7
            ops['bdreg'].value_constraints.append(
                    minmax_constraint(
                        what='index',
                        getint=lambda reg : reg.idx,
                        makeval=lambda idx : sve_vreg(reg_idx=idx),
                        minval=0, maxval=max_reg_idx
                    )
            )

        if mod.PART in mods:
            max_part = (adt_size(c_dt) // adt_size(a_dt)) - 1
            ops['part'] = osh(
                ot.IMMEDIATE, orl.PARAM, None, None,
                value_constraints=[minmax_constraint(minval=0, maxval=max_part)]
            )

        # widening needs SVE2, but BF16 into FP32 (bfmlalb) only BF16;
        # BF16 arithmetic (bfmla, bfadd) needs SVE_B16B16
        requires = {"SVE2"} if is_widening else set()
        if adt.BF16 in types:
            requires = {"BF16"} if is_widening else {"B16B16"}
        if types & {adt.FP8E4M3, adt.FP8E5M2}:
            requires.add("FP8FMA")
        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            requires=frozenset(requires),
            operands=ops
        ))

    for dt in _FLOATS:
        for m in base_mods:
            for om in base_opd_mods:
                if opd_mod.BLOCKLANE in om.get('bdreg',set()):
                    add_sig(dt, dt, dt, mods=m, opd_mods=om)
                else:
                    add_sig(dt, dt, dt, mods=m | {mod.MASK}, opd_mods=om)
                    # unpredicated (vectors): floating-point fadd and fmul
                    if unpredicated and not m:
                        add_sig(dt, dt, dt, mods=m, opd_mods=om)
                if dt in _WIDENING_2X_MAP:
                    add_sig(dt, dt, _WIDENING_2X_MAP[dt],
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)
                if dt in _WIDENING_4X_MAP:
                    add_sig(dt, dt, _WIDENING_4X_MAP[dt],
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)

    for dt in _SIGNED_INTS:
        for m in base_mods:
            for om in base_opd_mods:
                if opd_mod.BLOCKLANE in om.get('bdreg',set()):
                    add_sig(dt, dt, dt, mods=m, opd_mods=om)
                else:
                    add_sig(dt, dt, dt, mods=m | {mod.MASK}, opd_mods=om)
                if dt in _WIDENING_2X_MAP:
                    add_sig(dt, dt, _WIDENING_2X_MAP[dt],
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)
                if dt in _WIDENING_4X_MAP:
                    add_sig(dt, dt, _WIDENING_4X_MAP[dt],
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)

    for dt in _UNSIGNED_INTS:
        for m in base_mods:
            for om in base_opd_mods:
                # Widening only for unsigned ints
                if dt in _WIDENING_2X_MAP:
                    add_sig(dt, dt, _WIDENING_2X_MAP[dt],
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)
                if dt in _WIDENING_4X_MAP:
                    add_sig(dt, dt, _WIDENING_4X_MAP[dt],
                            mods=m | {mod.PART},
                            opd_mods=om,
                            is_widening=True)

    return sigs
