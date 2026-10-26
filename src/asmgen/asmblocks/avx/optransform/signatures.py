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
    optransform_modifier as mod
)

from ....registers import asm_data_type as adt


def make_avx_opd3_signatures(supports_np : bool, has_fp16 : bool) -> list[sig]:
    """
    Generate signatures for NEON opd3 operations
    """
    sigs = []

    base_mods = [set()]
    if supports_np:
        base_mods.extend([{mod.NP}])

    def add_sig(a_dt, b_dt, c_dt, *, mods):

        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, a_dt),
            'bdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, b_dt),
            'cdreg': osh(ot.REGISTER, orl.DATA, rt.VEC, c_dt)
        }
        sigs.append(sig(
            modifiers=mods,
            structural_params={},
            operands=ops,
            # the ph instructions: AVX512-FP16
            requires=frozenset({"FP16"}) if adt.FP16 in (a_dt, b_dt, c_dt) else frozenset()
        ))

    dts = [adt.FP64, adt.FP32]
    if has_fp16:
        dts.append(adt.FP16)
    for dt in dts:
        for m in base_mods:
            add_sig(dt, dt, dt, mods=m)


    return sigs
