# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for SVE opmem operations
"""

from dataclasses import dataclass
from typing import Type

from ...op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rt,
    opmem_modifier as mod,
    operand_modifier as opd_mod
)

from ...op.constraint import (
    minmax_constraint,
    multiple_constraint,
    otherplusnmod_constraint,
    regidx_constraint
)
from ...op.misc import make_ord_prefix as mop

from ..types import sve_vreg

from ....registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    adt_size
)

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

INDEX_ADT_SIZE_MAP = {
    8 : adt.SINT64,
    4 : adt.SINT32,
}
INDEX_AIT_SIZE_MAP = {
    8 : ait.INT64,
    4 : ait.INT32,
}

@dataclass(kw_only=True)
class sve_struct_constraint(regidx_constraint,otherplusnmod_constraint):
    """
    Constraint ensuring structured loads/stores use consecutive registers
    """
    reg_class : Type[sve_vreg] = sve_vreg
    offset : int = 1
    modval : int = 32

def make_sve_opmem_signatures(bcast_supported=False):
    """
    Generate signatures for SVE opmem operations

    :param bcast_supported: whether the instruction supports broadcasts (loads only)
    """
    sigs = []

    def add_sig(dt, *, mods, opd_mods=None, nstructs=1, idx_size=None):
        if opd_mods is None:
            opd_mods = {}

        ops = {
            'adreg': osh(ot.REGISTER, orl.DATA, rt.VEC, dt,
                         modifiers=opd_mods.get('adreg', set())),
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64,
                         modifiers=opd_mods.get('agreg', set())),
            'amreg': osh(ot.REGISTER, orl.MASK, rt.MASK, dt,
                         modifiers=opd_mods.get('amreg', set()))
        }

        struct_params={}
        if opd_mod.BLOCK in opd_mods.get('adreg', set()):
            # 128-bit blocks (ld1rq)
            struct_params['adreg_blocksize'] = 16 // adt_size(dt)

        # structured ld/st logic
        if mod.STRUCT in mods:
            struct_params['nstructs'] = nstructs
            for i in range(1, nstructs):
                reg_name = f"{mop(i)}dreg" # bdreg, cdreg, ddreg
                ops[reg_name] = osh(
                    ot.REGISTER, orl.DATA, rt.VEC, dt,
                    modifiers=opd_mods.get(reg_name, set()),
                    value_constraints=[
                        sve_struct_constraint(other=f"{mop(i-1)}dreg")
                    ])

        if mod.VINDEX in mods:
            # SVE gathers/scatters
            idx_sz = idx_size or adt_size(dt)
            # Fallback to 32-bit indices if size < 4 to prevent KeyError
            # if the >= 4 filter is removed
            idx_dt = INDEX_ADT_SIZE_MAP.get(idx_sz, adt.SINT32)
            idx_it = INDEX_AIT_SIZE_MAP.get(idx_sz, ait.INT32)

            ops['vidxreg'] = osh(ot.REGISTER, orl.ADDRESS, rt.VEC, idx_dt,
                                 modifiers=opd_mods.get('vidxreg', set()))
            struct_params['it'] = idx_it

        # gathers and scatters [z.d, #imm]: 0..31 elements; ld1rq [x, #imm]:
        # -128..112 bytes in multiples of 16
        if mod.IOFFSET in mods and opd_mod.BLOCK not in opd_mods.get('adreg', set()) \
                and opd_mod.BCAST in opd_mods.get('adreg', set()):
            # ld1r{b,h,w,d} [x, #imm]: 0..63 elements
            ops['ioffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[
                                     minmax_constraint(minval=0, maxval=63 * adt_size(dt)),
                                     multiple_constraint(multiple=adt_size(dt))])
        elif mod.IOFFSET in mods and opd_mod.BLOCK in opd_mods.get('adreg', set()):
            ops['ioffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[
                                     minmax_constraint(minval=-128, maxval=112),
                                     multiple_constraint(multiple=16)])
        elif mod.IOFFSET in mods:
            ops['ioffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[
                                     minmax_constraint(minval=0, maxval=31 * adt_size(dt)),
                                     multiple_constraint(multiple=adt_size(dt))])
        if mod.GOFFSET in mods:
            ops['offreg'] = osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.SINT64,
                                modifiers=opd_mods.get('offreg', set()))
        # [x, #imm, MUL VL]
        if mod.VOFFSET in mods:
            # structures: whole ones, -8..7 of them (ld2d [x, #-16..14, MUL VL])
            n = nstructs if mod.STRUCT in mods else 1
            ops['voffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[minmax_constraint(minval=-8 * n, maxval=7 * n)]
                                 + ([multiple_constraint(multiple=n)] if n > 1 else []))

        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops
        ))

    for dt in _FLOATS+_INTS:
        # Base predicated load/store
        add_sig(dt, mods={mod.MASK})

        # Gathers/Scatters
        if adt_size(dt) >= 4:
            # TODO: Handle size < 4
            #       (requires unpacking loops/multiple gathers as SVE only
            # provides 32-bit or 64-bit index elements).
            add_sig(dt, mods={mod.MASK, mod.VINDEX})
            if adt_size(dt) == 8:
                # 32-bit offsets in 64-bit lanes, sign-extended
                add_sig(dt, mods={mod.MASK, mod.VINDEX}, idx_size=4)
            add_sig(dt, mods={mod.MASK, mod.VINDEX, mod.IOFFSET})

        # Address modification
        add_sig(dt, mods={mod.MASK, mod.VOFFSET})
        add_sig(dt, mods={mod.MASK, mod.GOFFSET})

        if bcast_supported:
            # ld1r (load scalar and broadcast to all lanes)
            add_sig(dt, mods={mod.MASK}, opd_mods={'adreg': {opd_mod.BCAST}})
            add_sig(dt, mods={mod.MASK, mod.IOFFSET}, opd_mods={'adreg': {opd_mod.BCAST}})
            # a 128-bit block replicated into every block: ld1rq
            add_sig(dt, mods={mod.MASK}, opd_mods={'adreg': {opd_mod.BLOCK, opd_mod.BCAST}})
            add_sig(dt, mods={mod.MASK, mod.IOFFSET},
                    opd_mods={'adreg': {opd_mod.BLOCK, opd_mod.BCAST}})

        for nstructs in range(2,5):
            # ld2/ld3/ld4 - [x], [x, #imm, MUL VL], [x, x, lsl #e]: a loop
            # over a panel's structures advances by whole ones
            add_sig(dt, mods={mod.MASK, mod.STRUCT}, nstructs=nstructs)
            add_sig(dt, mods={mod.MASK, mod.STRUCT, mod.VOFFSET}, nstructs=nstructs)
            add_sig(dt, mods={mod.MASK, mod.STRUCT, mod.GOFFSET}, nstructs=nstructs)

            if bcast_supported:
                # ld2r/ld3r/ld4r (load structured scalars and broadcast
                # to respective vectors)
                bcast_mods = {'adreg': {opd_mod.BCAST}}
                for i in range(1, nstructs):
                    bcast_mods[f"{mop(i)}dreg"] = {opd_mod.BCAST}

                add_sig(dt, mods={mod.MASK, mod.STRUCT},
                        opd_mods=bcast_mods, nstructs=nstructs)

    return sigs
