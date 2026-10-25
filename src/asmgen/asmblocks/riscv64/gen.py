# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V 64bit asm generator and related types
"""

from copy import deepcopy

from ...registers import (
    adt_size,

    asm_data_type as adt,
    greg_base, freg_base
)

from ..noarch import asmgen,comparison

from .types import FCDT_SUFFIXES, FDT_SUFFIXES
from .types import riscv64_freg, riscv64_greg
from ...callconv.callconv import callconv


from .opmem import riscv64_load,riscv64_store
from .opmem import riscv64_prefetch
from ...registers import asm_data_type as _ADT
from ..op.operand import operand_modifier as _opd_mod
from ..op.move import move_modifier as _move_mod
from ..op.scalar import BITS_GP as _BITS_GP
from ..op.opset import opset_modifier as _opset_mod
from ..op.opd3 import opd3_modifier as _opd3_mod

_GP = _ADT.UINT64  # general registers' arithmetic: 64-bit


# pylint: disable=too-many-public-methods

class riscv64(asmgen):
    """
    RISC-V 64bit asmgen implementation
    """

    fdt_suffixes = FDT_SUFFIXES

    fcdt_suffixes = FCDT_SUFFIXES


    cb_insts = {
            'NZ' : 'bne',
            'EZ' : 'beq',
            'NE' : 'bne',
            'EQ' : 'beq',
            'LE' : 'ble',
            'GE' : 'bge',
            'LT' : 'blt',
            'GT' : 'bgt',
            }

    def __init__(self):
        super().__init__()
        self.callconvs = {
            "rvg" : callconv(
                param_regs={ 'greg' : list(range(18, 26)),
                             'freg' : list(range(10, 18))},
                caller_save_lists={ 'greg' : [26, # ra
                                              0, 1, 2, 3 , 4, 5, 6, # t0-t6
                                              18, 19, 20, 21, 22, 23, 24, 25 # a0-a7
                                              ],
                                    'freg' : [0, 1, 2, 3, 4, 5, 6, 7,
                                              10, 11, 12, 13, 14, 15, 16, 17,
                                              28, 29, 30, 31]},
                callee_save_lists={ 'greg' : [27, #sp
                                              30, #s0
                                              7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 #s1-s11
                                             ],
                                    'freg' : [8, 9,
                                              18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
                spreg=27)
            }
        self.default_callconv = "rvg"

        self.load = riscv64_load(asmwrap=self.asmwrap)
        self.store = riscv64_store(asmwrap=self.asmwrap)
        self.prefetch = riscv64_prefetch(asmwrap=self.asmwrap)

    def create_callconv(self, name : str = "default"):

        if "default" == name:
            name = self.default_callconv

        return deepcopy(self.callconvs[name])

    def greg(self, reg_idx : int) -> greg_base:
        return riscv64_greg(reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        _ = dt # explicitly unused
        return riscv64_freg(reg_idx)

    @property
    def are_fregs_in_vregs(self) -> bool:
        return False

    def label(self, *, label : str) -> str:
        return self.asmwrap(f"{self.labelstr(label)}:")

    def jump(self, *, label : str) -> str:
        return self.asmwrap(f"j {self.labelstr(label)}")

    def fcb(self, *, reg1, reg2, cmp: comparison, label: str, dt: adt, greg=None) -> str:
        """
        cb on FP registers: equality as floating point, so -0.0 equals 0.0
        and NaN equals nothing
        """
        if reg2 is None or greg is None:
            raise ValueError("RISC-V compares FP registers into a general register: "
                             "pass reg2 and greg")
        lab = self.labelstr(label)
        out = self.asmwrap(f"feq.{self.fcdt_suffixes[dt]} {greg},{reg1},{reg2}")
        if cmp.name in ('EQ', 'EZ'):
            return out + self.asmwrap(f"bnez {greg},{lab}")
        if cmp.name in ('NE', 'NZ'):
            return out + self.asmwrap(f"beqz {greg},{lab}")
        raise NotImplementedError(f"RISC-V FP comparison {cmp.name}")

    def mov_greg_to_freg(self, *, src: greg_base, dst, dt: adt) -> str:
        """
        Move a general register's low bits into an FP register: an FP
        constant from its bit pattern
        """
        return self.move(dregs=[src, dst], dts=[_BITS_GP[dt], dt], modifiers={_move_mod.BITS})

    def cb(self, *, reg1, reg2, cmp: comparison, label: str,
           dt: adt | None = None, greg: greg_base | None = None) -> str:
        if isinstance(reg1, freg_base):
            return self.fcb(reg1=reg1, reg2=reg2, cmp=cmp, label=label, dt=dt, greg=greg)
        inst = self.cb_insts[cmp.name]
        if reg2 is None:
            return self.asmwrap(f"{inst} {reg1},zero,{self.labelstr(label)}")

        return self.asmwrap(f"{inst} {reg1},{reg2},{self.labelstr(label)}")

    def jzero(self, *, reg : greg_base, label : str) -> str:
        return self.asmwrap(f" beq {reg},zero,{self.labelstr(label)}")

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        cdt_suf = self.fcdt_suffixes[dt]
        asmblock  = self.asmwrap(f"fmv.{dt_suf}.x {freg2},zero")
        asmblock += self.asmwrap(f"feq.{cdt_suf} {greg},{freg1},{freg2}")
        asmblock += self.asmwrap(f"bnez {greg},{self.labelstr(label)}")
        return asmblock

    def loopbegin(self, *, reg : greg_base, label : str) -> str:
        asmblock  = self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopbegin_nz(self, *, reg : greg_base, label : str, labelskip : str) -> str:
        asmblock  = self.asmwrap(f"beq {reg},zero,{self.labelstr(labelskip)}")
        asmblock += self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopend(self, *, reg : greg_base, label : str) -> str:
        return self.asmwrap(f"bnez {reg},{self.labelstr(label)}")
    def max_gregs(self) -> int:
        return len(riscv64_greg.names)
    def max_mregs(self):
        return 0
    def max_fregs(self) -> int:
        return len(riscv64_freg.names)


    def kiterkleft_pow2(self, *, kreg : greg_base,
                        kleftreg : greg_base,
                        unroll : int) -> str:
        """
        kiterkleft variant for unroll values that are powers of 2
        
        Can be simplified down to a "right shift" and "and"
        """

        asmblock = ""

        unrollpower = unroll.bit_length()-1

        # Example unroll=4: kleft = k & 0b0111
        asmblock += self.asmwrap(f"andi {kleftreg},{kreg},{unroll-1}")
        # Example unroll=4: k = k >> 2
        asmblock += self.shift_greg_right(reg=kreg,bit_count=unrollpower)

        return asmblock

    def kiterkleft(self, *, kreg : greg_base,
                   kleftreg : greg_base,
                   tmpreg : greg_base,
                   unroll : int) -> str:

        if unroll.bit_count() == 1:
            return self.kiterkleft_pow2(kreg=kreg,kleftreg=kleftreg,unroll=unroll)

        # TODO: Division by invariant multiplication


        # DIV version

        asmblock = ""

        asmblock += self.mov_greg_imm(reg=tmpreg, imm=unroll)
        # kleft = k % unroll
        asmblock += self.asmwrap(f"remu {kleftreg},{kreg},{tmpreg}")
        # tmp = k // unroll
        asmblock += self.asmwrap(f"divu {kreg},{kreg},{tmpreg}")


        return asmblock

    def mov_freg(self, *, src : freg_base, dst : freg_base, dt : adt) ->  str:
        # fmv.s / fmv.d (fsgnj): fmv.w is no instruction - fmv.w.x moves an
        # integer register
        return self.move(dregs=[src, dst], dts=[dt, dt])

    def mov_greg(self, *, src : greg_base, dst : greg_base) -> str:
        # There is no mov instruction?
        return self.move(dregs=[src, dst], dts=[_GP, _GP])

    def mov_param_to_greg_shift(self, *, param : str, dst : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"slli {dst},%[{param}],{bit_count}")

    def mov_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.set(dreg=reg, dt=_ADT.UINT64, modifiers={_opset_mod.CONST}, value=imm)

    def mul_greg_imm(self, *, src : greg_base, dst : greg_base, factor : int) -> str:
        assert src != dst
        # no immediate form: the factor into dst first
        asmblock = self.mov_greg_imm(reg=dst, imm=factor)
        asmblock += self.mul(adreg=src, bdreg=dst, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)
        return asmblock

    def mul_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        return self.mul(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)

    def add_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.add(adreg=reg, bimm=imm, cdreg=reg, a_dt=_GP, c_dt=_GP)

    def add_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        return self.add(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)

    def sub_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        return self.sub(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)

    def add_greg_greg_shifted(self, *, dst : greg_base, reg1 : greg_base,
                              reg2 : greg_base, shift : int) -> str:
        try:
            return self.add(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP,
                          operand_modifiers={'bdreg': {_opd_mod.ISHIFT}}, bdreg_shift=shift)
        except ValueError as err:
            # no form (or not with these features): what callers probing it expect
            raise NotImplementedError(f"RISC-V has no add of a register shifted by {shift} "
                                      f"in one instruction here: {err}") from err

    def sub_greg_greg_shifted(self, *, dst : greg_base, reg1 : greg_base,
                              reg2 : greg_base, shift : int) -> str:
        try:
            return self.sub(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP,
                          operand_modifiers={'bdreg': {_opd_mod.ISHIFT}}, bdreg_shift=shift)
        except ValueError as err:
            # no form (or not with these features): what callers probing it expect
            raise NotImplementedError(f"RISC-V has no sub of a register shifted by {shift} "
                                      f"in one instruction here: {err}") from err

    def shift_greg_left(self, *, reg : greg_base, bit_count : int) -> str:
        return self.shift(adreg=reg, bimm=bit_count, cdreg=reg, a_dt=_GP, c_dt=_GP, modifiers={_opd3_mod.SHL})

    def shift_greg_right(self, *, reg : greg_base, bit_count : int) -> str:
        return self.shift(adreg=reg, bimm=bit_count, cdreg=reg, a_dt=_GP, c_dt=_GP, modifiers={_opd3_mod.SHR})

    def zero_greg(self, *, greg : greg_base) -> str:
        return self.set(dreg=greg, dt=_ADT.UINT64, modifiers={_opset_mod.CONST}, constant="zero")

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        # a conversion's format is the arithmetic one: fcvt.s.w, not fcvt.w.w
        return self.set(dreg=freg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def reserved_registers(self) -> dict[str, dict[str, int]]:
        # indices as riscv64_greg names them: ra, sp, gp, tp, s0 (fp), zero
        return {"greg": {"ra": 26, "sp": 27, "gp": 28, "tp": 29, "fp": 30, "zero": 31}}
