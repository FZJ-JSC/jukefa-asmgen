# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for NEON opmem operations
"""

from dataclasses import dataclass
from typing import Type

from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    operand_modifier as opd_mod,
    register_type as rt,
    opmem_modifier as mod
)

from ...op.constraint import (
    otherplusnmod_constraint,
    minmax_constraint,
    multiple_constraint,
    oneof_constraint,
    either_constraint,
    regidx_constraint
)
from ...op.misc import make_ord_prefix as mop

from ..types import neon_vreg

from ....registers import (
    asm_data_type as adt,
    adt_size
)

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

@dataclass(kw_only=True)
class neon_struct_constraint(regidx_constraint,otherplusnmod_constraint):
    """
    Constraint ensuring structured loads/stores use consecutive registers
    """
    reg_class : Type[neon_vreg] = neon_vreg
    offset : int = 1
    modval : int = 32

def make_neon_opmem_signatures(bcast_supported=False):
    """
    Generate signatures for NEON opmem operations

    :param bcast_supported: whether the instruction supports broadcasts (loads only)
    """
    sigs = []

    def add_sig(dt, *, mods, opd_mods, nstructs=1, postinc_reg=False):
        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
        }

        struct_params={}

        # structured ld/st logic
        if mod.STRUCT in mods:
            struct_params['nstructs'] = nstructs
            for i in range(1, nstructs):
                ops[f"{mop(i)}dreg"] = osh(
                    ot.REGISTER, orl.DATA, rt.VEC, dt,
                    value_constraints=[
                        neon_struct_constraint(other=f"{mop(i-1)}dreg")
                    ])

        # ldr/str q: an unscaled offset within -256..255 (the assembler encodes
        # ldur/stur) or a multiple of 16 up to 65520; voffset counts vectors
        if mod.IOFFSET in mods:
            ops['ioffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[either_constraint(options=[
                                     [minmax_constraint(minval=-256, maxval=255)],
                                     [minmax_constraint(minval=0, maxval=65520),
                                      multiple_constraint(multiple=16)]])])
        if mod.VOFFSET in mods:
            ops['voffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[minmax_constraint(minval=-16, maxval=4095)])

        for opd, omods in opd_mods.items():
            ops[opd].modifiers = omods
            if opd_mod.ILANE in omods:
                max_lane = (16 // adt_size(dt))-1
                ops[f"{opd}_lane"] = osh(
                        ot.IMMEDIATE, orl.PARAM, None, None,
                        value_constraints=[
                            minmax_constraint(minval=0,maxval=max_lane)]
                        )

        if mod.POSTINC in mods:
            if postinc_reg:
                ops['increg'] = osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64)
            else:
                # ld1, ld1r, ld2...: the immediate post-increment is exactly the
                # bytes transferred - an element per structure for broadcasts and
                # lanes, a 16-byte register per structure otherwise
                adreg_mods = opd_mods.get('adreg', set())
                per_struct = adt_size(dt) if {opd_mod.BCAST, opd_mod.ILANE} & adreg_mods else 16
                ops['iinc'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                  value_constraints=[oneof_constraint(
                                      valset=[per_struct * nstructs])])

        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops
        ))

    for dt in _FLOATS+_INTS:
        add_sig(dt, mods=set(), opd_mods=dict())
        add_sig(dt, mods={mod.IOFFSET}, opd_mods=dict())
        add_sig(dt, mods={mod.VOFFSET}, opd_mods=dict())
        add_sig(dt, mods=set(), opd_mods={'adreg': {opd_mod.ILANE}})
        add_sig(dt, mods={mod.VOFFSET}, opd_mods={'adreg': {opd_mod.ILANE}})
        add_sig(dt, mods={mod.POSTINC}, opd_mods=dict(), postinc_reg=False)
        add_sig(dt, mods={mod.POSTINC}, opd_mods=dict(), postinc_reg=True)
        if bcast_supported:
            add_sig(dt, mods=set(), opd_mods={'adreg': {opd_mod.BCAST}})
            # No BCAST with offsets, but POSTINC is allowed
            add_sig(dt, mods={mod.POSTINC},
                    opd_mods={'adreg': {opd_mod.BCAST}}, postinc_reg=False)
            add_sig(dt, mods={mod.POSTINC},
                    opd_mods={'adreg': {opd_mod.BCAST}}, postinc_reg=True)
        for nstructs in range(2,5):
            opd_bcast_mods = {
                    f"{mop(i)}dreg": {opd_mod.BCAST} for i in range(nstructs)}
            opd_ilane_mods = {
                    f"{mop(i)}dreg": {opd_mod.ILANE} for i in range(nstructs)}

            add_sig(dt, mods={mod.STRUCT}, opd_mods=dict(), nstructs=nstructs)
            add_sig(dt, mods={mod.STRUCT},
                    opd_mods=opd_ilane_mods,
                    nstructs=nstructs)
            # There is also a STRUCT+ILANE+IOFFSET, but I don't get what the
            # constraints on the immediate offset are, it seems like
            # just one value for each data type?
            add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                    opd_mods=dict(),
                    nstructs=nstructs, postinc_reg=False)
            add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                    opd_mods=dict(),
                    nstructs=nstructs, postinc_reg=True)
            add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                    opd_mods=opd_ilane_mods,
                    nstructs=nstructs, postinc_reg=True)
            if bcast_supported:
                add_sig(dt, mods={mod.STRUCT},
                        opd_mods = opd_bcast_mods,
                        nstructs=nstructs)
                add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                        opd_mods = opd_bcast_mods,
                        nstructs=nstructs, postinc_reg=False)
                add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                        opd_mods = opd_bcast_mods,
                        nstructs=nstructs, postinc_reg=True)

    return sigs
