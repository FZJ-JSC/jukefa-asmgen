# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for SME opmem operations
"""

from dataclasses import dataclass,field
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
    otherplusnmod_constraint,
    minmax_constraint,
    oneof_constraint,
    regidx_constraint
)
from ...op.misc import make_ord_prefix as mop

from ...sve.types import sve_vreg,sve_preg
from ...aarch64.types import aarch64_greg

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
class sme_rowcolreg_constraint(regidx_constraint, minmax_constraint):
    """
    Constraint ensuring row/col registers have an index between 12-15
    """
    reg_class : Type[aarch64_greg] = aarch64_greg
    minval : int = 12
    maxval : int = 15

@dataclass(kw_only=True)
class sme_nt_struct_constraint(regidx_constraint, otherplusnmod_constraint):
    """
    Constraint ensuring correct dreg register index offset for nt struct ld/st
    """
    reg_class : Type[sve_vreg] = sve_vreg
    offset : int = field(init=False)
    modval : int = 32
    nstructs : int

    def __post_init__(self):

        if self.nstructs == 2:
            self.offset = 8
        if self.nstructs == 4:
            self.offset = 4

        super().__post_init__()

@dataclass(kw_only=True)
class sme_nt_struct_firstreg_constraint(regidx_constraint, oneof_constraint):
    """
    Constraint ensuring the first register in an nt struct ld/st is bounded correctly
    """
    reg_class : Type[sve_vreg] = sve_vreg
    valset : int = field(init=False)
    nstructs : int

    def __post_init__(self):

        if self.nstructs == 2:
            self.valset = set(range(8)).union(set(range(16,24)))
        if self.nstructs == 4:
            self.valset = set(range(4)).union(set(range(16,20)))

        super().__post_init__()

@dataclass(kw_only=True)
class sme_mreg_pn_constraint(regidx_constraint, minmax_constraint):
    """
    Constraint ensuring the predicate is a pn register
    """
    reg_class : Type[sve_preg] =  sve_preg
    minval : int = 8
    maxval : int = 15

@dataclass(kw_only=True)
class sme_mreg_notpn_constraint(regidx_constraint, minmax_constraint):
    """
    Constraint ensuring the predicate is a pn register
    """
    reg_class : Type[sve_preg] =  sve_preg
    minval : int = 0
    maxval : int = 7

def make_sme_opmem_signatures():
    """
    Generate signatures for SME opmem operations

    """
    sigs = []

    def add_sig(dt, *, mods, opd_mods=None, nstructs=1):
        ops = {
            'agreg': osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.UINT64),
            'amreg': osh(ot.REGISTER, orl.MASK, rt.MASK, dt)
        }
        if opd_mods is None:
            opd_mods = dict()

        struct_params={}

        # structured ld/st logic
        if mod.STRUCT in mods:
            ops['amreg'].value_constraints.append(
                    sme_mreg_pn_constraint()
                    )
            struct_params['nstructs'] = nstructs
            ops['adreg'] = osh(ot.REGISTER, orl.DATA, rt.VEC, dt,
                               value_constraints = [
                                   sme_nt_struct_firstreg_constraint(nstructs=nstructs)
                                ])
            for i in range(1, nstructs):
                ops[f"{mop(i)}dreg"] = osh(
                    ot.REGISTER, orl.DATA, rt.VEC, dt,
                    value_constraints=[
                        sme_nt_struct_constraint(other=f"{mop(i-1)}dreg",nstructs=nstructs)
                    ])
        else:
            ops['adreg'] = osh(ot.REGISTER, orl.DATA, rt.TILE, dt)
            ops['amreg'].value_constraints.append(
                    sme_mreg_notpn_constraint()
                    )

        for name, omods in opd_mods.items():
            ops[name].modifiers = omods

        if opd_mod.ROW in opd_mods.get('adreg',set()):
            ops['adreg_rowreg'] = osh(ot.REGISTER, orl.PARAM, rt.GP, adt.SINT32,
                                      value_constraints=[sme_rowcolreg_constraint()])
            maxrow = 16//adt_size(dt) - 1
            ops['adreg_immrow'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                      value_constraints=[
                                          minmax_constraint(minval=0,maxval=maxrow)])

        if opd_mod.COL in opd_mods.get('adreg',set()):
            ops['adreg_colreg'] = osh(ot.REGISTER, orl.PARAM, rt.GP, adt.SINT32,
                                value_constraints=[sme_rowcolreg_constraint()])
            maxrow = 16//adt_size(dt) - 1
            ops['adreg_immcol'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                value_constraints=[minmax_constraint(minval=0,maxval=maxrow)])

        if mod.GOFFSET in mods:
            ops['offreg'] = osh(ot.REGISTER, orl.ADDRESS, rt.GP, adt.SINT64)
        if mod.VOFFSET in mods:
            # [x, #imm, MUL VL]: as SVE's contiguous forms, -8..7 vectors
            ops['voffset'] = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                                 value_constraints=[minmax_constraint(minval=-8, maxval=7)])

        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops
        ))

    for dt in _FLOATS+_INTS:
        # LD1B/H/W/D { <ZAt><HV>.B/H/S/D[<Ws>, <offs>] }, <Pg>/Z, [<Xn|SP>{, <Xm>, LSL #2}]
        add_sig(dt, mods={mod.MASK}, opd_mods={'adreg':{opd_mod.ROW}})
        add_sig(dt, mods={mod.MASK}, opd_mods={'adreg':{opd_mod.COL}})
        add_sig(dt, mods={mod.MASK, mod.GOFFSET}, opd_mods={'adreg':{opd_mod.ROW}})
        add_sig(dt, mods={mod.MASK, mod.GOFFSET}, opd_mods={'adreg':{opd_mod.COL}})
        # LD1NT...
        for nstructs in [2,4]:
            add_sig(dt, mods={mod.MASK, mod.NT, mod.STRUCT}, nstructs=nstructs)
            add_sig(dt, mods={mod.MASK, mod.NT, mod.STRUCT, mod.GOFFSET}, nstructs=nstructs)
            add_sig(dt, mods={mod.MASK, mod.NT, mod.STRUCT, mod.VOFFSET}, nstructs=nstructs)

    return sigs
