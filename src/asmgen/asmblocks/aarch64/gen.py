# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
ARM64/AArch64 asm generator and related types
"""

from copy import deepcopy
from typing import Union

from ..noarch import compiler_family
from ..noarch import asmgen,comparison
from ...registers import (
    adt_size,

    asm_data_type as adt,
    greg_base, freg_base
)

from .types import aarch64_greg,aarch64_freg
from ...callconv.callconv import callconv

from .opmem import aarch64_load, aarch64_store
from .opmem import aarch64_prefetch
from ...registers import asm_data_type as _ADT
from ..op.operand import operand_modifier as _opd_mod
from ..op.move import move_modifier as _move_mod
from ..op.scalar import BITS_GP as _BITS_GP
from ..op.opset import opset_modifier as _opset_mod
from ..op.opd3 import opd3_modifier as _opd3_mod

_GP = _ADT.UINT64  # general registers' arithmetic: 64-bit

class aarch64(asmgen):
    """
    ARM64/AArch64 asmgen implementation
    """

    #: -march's base, and the features as its extensions (per backend)
    MARCH_BASE = "armv8-a"
    MARCH_EXTENSIONS : dict[str, str] = {}

    def compiler_flags(self, compiler : str = "gcc") -> list[str]:
        compiler_family(compiler)
        enabled = self.enabled_features()
        extensions = [self.MARCH_EXTENSIONS[f] for f in self.FEATURES
                      if f in enabled and f in self.MARCH_EXTENSIONS]
        return ["-march=" + "+".join([self.MARCH_BASE] + extensions)]

    #: the nop: nop
    NOP_BYTES = 4
    NOP_ENCODING = 0xd503201f

    dt_greg_pfx = {
            adt.DOUBLE : "x",
            adt.SINGLE : "w",
            }

    cb_insts = {
            'NZ' : 'cbnz',
            'EZ' : 'cbz',
            'NE' : 'b.ne',
            'EQ' : 'b.eq',
            'LE' : 'b.le',
            'GE' : 'b.ge',
            'LT' : 'b.lt',
            'GT' : 'b.gt',
            }

    def __init__(self):
        super().__init__()
        self.callconvs = {
            "aapcs64" : callconv(
                param_regs={ 'greg' : list(range(0, 8)),
                             'freg' : list(range(0, 8))},
                caller_save_lists={ 'greg' : list(range(0, 8)) + # func args/return vals
                                             [8] + # XR, indirect result reg
                                             list(range(9,16)) + # locals
                                             [18] # platform reg
                                        ,
                                    'freg' : list(range(0, 8)) +
                                             list(range(16,32))
                                   },
                callee_save_lists={ 'greg' : list(range(19,29)) + # Callee-saved
                                             [29, # Frame Pointer
                                              30  # Link Register
                                              ] +
                                             [31], # SP
                                    'freg' : list(range(8,16))},
                spreg=31)
            }
        self.load = aarch64_load(asmwrap=self.asmwrap)
        self.store = aarch64_store(asmwrap=self.asmwrap)
        self.prefetch = aarch64_prefetch(asmwrap=self.asmwrap)
        self.default_callconv = "aapcs64"

    def create_callconv(self, name : str = "default"):

        if "default" == name:
            name = self.default_callconv

        return deepcopy(self.callconvs[name])

    @property
    def are_fregs_in_vregs(self) -> bool:
        return True

    def label(self, *, label : str) -> str:
        return self.asmwrap(f"{self.labelstr(label)}:")

    def jump(self, *, label : str) -> str:
        return self.asmwrap(f"b {self.labelstr(label)}")

    def loopbegin(self, *, reg : greg_base,
                  label : str) -> str:
        asmblock  = self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub {reg},{reg},1")
        return asmblock

    def loopbegin_nz(self, *, reg : greg_base,
                     label : str,
                     labelskip : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.eq {self.labelstr(labelskip)}")
        asmblock += self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub {reg},{reg},1")
        return asmblock

    def loopend(self, *, reg : greg_base,
                label : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.ne {self.labelstr(label)}")
        return asmblock

    def fcb(self, *, reg1, reg2, cmp: comparison, label: str, dt: adt, greg=None) -> str:
        """
        cb on FP registers: equality as floating point, so -0.0 equals 0.0
        and NaN equals nothing
        """
        insts = {'EQ': 'b.eq', 'EZ': 'b.eq', 'NE': 'b.ne', 'NZ': 'b.ne',
                 'LT': 'b.mi', 'LE': 'b.ls', 'GT': 'b.gt', 'GE': 'b.ge'}
        other = reg2 if reg2 is not None else "#0.0"
        return self.asmwrap(f"fcmp {reg1},{other}") + \
            self.asmwrap(f"{insts[cmp.name]} {self.labelstr(label)}")

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
            return self.asmwrap(f"{inst} {reg1},{self.labelstr(label)}")

        return self.asmwrap(f"cmp {reg1},{reg2}")+\
            self.asmwrap(f"{inst} {self.labelstr(label)}")

    def jzero(self, *, reg : greg_base,
              label : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.eq {self.labelstr(label)}")
        return asmblock

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt):
        asmblock  = self.asmwrap(f"fcmp {freg1},#0.0")
        asmblock += self.asmwrap(f"b.eq {self.labelstr(label)}")
        return asmblock

    def shared_registers(self) -> list[tuple[str, str, dict[int, int] | None]]:
        # a scalar FP register is the low part of the vector register
        return [("freg", "vreg", None)]

    def reserved_registers(self) -> dict[str, dict[str, int]]:
        # x18 is the platform register on Apple and Windows
        return {"greg": {"sp": 31, "lr": 30, "fp": 29, "platform": 18}}
    def max_gregs(self):
        return 32
    def max_mregs(self):
        return 0
    def max_fregs(self):
        return 32

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
        asmblock += self.asmwrap(f"and {kleftreg},{kreg},#{unroll-1}")
        # Example unroll=4: k = k >> 2
        asmblock += self.shift_greg_right(reg=kreg,bit_count=unrollpower)

        return asmblock

    def kiterkleft(self, *, kreg : greg_base,
                   kleftreg : greg_base,
                   tmpreg : greg_base,
                   unroll : int) -> str:

        if unroll.bit_count() == 1:
            return self.kiterkleft_pow2(kreg=kreg,kleftreg=kleftreg,unroll=unroll)

        # TODO: Division by invariant multiplication (the naive attempt below fails,
        #       need to read up on it)

        # runroll = (1<<64) // unroll
        # shift   = 64 - unroll.bit_length()
        # asmblock = ""

        # # Move reciprocal into tmpreg
        # asmblock += self.asmwrap(f"mov  {tmpreg}, #{(runroll >>  0) & 0xFFFF}")
        # asmblock += self.asmwrap(f"movk {tmpreg}, #{(runroll >> 16) & 0xFFFF}, lsl 16")
        # asmblock += self.asmwrap(f"movk {tmpreg}, #{(runroll >> 32) & 0xFFFF}, lsl 32")
        # asmblock += self.asmwrap(f"movk {tmpreg}, #{(runroll >> 48) & 0xFFFF}, lsl 48")


        # # tmp = k // unroll
        # asmblock += self.mul_greg_greg(dst=tmpreg, reg1=kreg, reg2=tmpreg)
        # asmblock += self.shift_greg_right(reg=tmpreg,bit_count=shift)
        #
        # # kleft = k - (tmp * unroll)
        # asmblock += self.mov_greg_imm(reg=kleftreg, imm=unroll)
        # asmblock += self.asmwrap(f"msub {kleftreg},{tmpreg},{kleftreg},{kreg}")

        # # k = tmp
        # asmblock += self.mov_greg(src=tmpreg,dst=kreg)

        # return asmblock


        # DIV version

        asmblock = ""

        asmblock += self.mov_greg_imm(reg=kleftreg, imm=unroll)
        # tmp = k // unroll
        asmblock += self.asmwrap(f"udiv {tmpreg},{kreg},{kleftreg}")
        # kleft = k - (tmp * unroll)
        asmblock += self.asmwrap(f"msub {kleftreg},{tmpreg},{kleftreg},{kreg}")
        # k = tmp
        asmblock += self.mov_greg(src=tmpreg,dst=kreg)


        return asmblock



    def mov_greg(self, *, src : greg_base, dst : greg_base) -> str:
        return self.move(dregs=[src, dst], dts=[_GP, _GP])

    def mov_freg(self, *, src : freg_base, dst : freg_base,
                 dt : adt) -> str:
        return self.move(dregs=[src, dst], dts=[dt, dt])

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        # fmov's immediate can't encode 0.0: the zero register can
        return self.set(dreg=freg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def mov_param_to_greg_shift(self, *, param : str,
                                dst : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"ldr {dst},%[{param},LSL#{bit_count}]")

    def mov_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.set(dreg=reg, dt=_ADT.UINT64, modifiers={_opset_mod.CONST}, value=imm)

    def mul_greg_imm(self, *, src : greg_base, dst : greg_base, factor):
        assert src != dst
        # no immediate form: the factor into dst first
        asmblock = self.mov_greg_imm(reg=dst, imm=factor)
        asmblock += self.mul(adreg=src, bdreg=dst, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)
        return asmblock

    def mul_greg_greg(self, *, dst : greg_base,
                      reg1 : greg_base, reg2 : greg_base) -> str:
        return self.mul(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)

    def add_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.add(adreg=reg, bimm=imm, cdreg=reg, a_dt=_GP, c_dt=_GP)

    def add_greg_greg(self, *, dst : greg_base,
                      reg1 : greg_base, reg2 : greg_base) -> str:
        return self.add(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)

    def sub_greg_greg(self, *, dst : greg_base,
                      reg1 : greg_base, reg2 : greg_base) -> str:
        return self.sub(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP)

    def add_greg_greg_shifted(self, *, dst : greg_base, reg1 : greg_base,
                              reg2 : greg_base, shift : int) -> str:
        try:
            return self.add(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP,
                          operand_modifiers={'bdreg': {_opd_mod.ISHIFT}}, bdreg_shift=shift)
        except ValueError as err:
            # no form (or not with these features): what callers probing it expect
            raise NotImplementedError(f"AArch64 has no add of a register shifted by {shift} "
                                      f"in one instruction here: {err}") from err

    def sub_greg_greg_shifted(self, *, dst : greg_base, reg1 : greg_base,
                              reg2 : greg_base, shift : int) -> str:
        try:
            return self.sub(adreg=reg1, bdreg=reg2, cdreg=dst, a_dt=_GP, b_dt=_GP, c_dt=_GP,
                          operand_modifiers={'bdreg': {_opd_mod.ISHIFT}}, bdreg_shift=shift)
        except ValueError as err:
            # no form (or not with these features): what callers probing it expect
            raise NotImplementedError(f"AArch64 has no sub of a register shifted by {shift} "
                                      f"in one instruction here: {err}") from err

    def shift_greg_left(self, *, reg : greg_base, bit_count : int) -> str:
        return self.shift(adreg=reg, bimm=bit_count, cdreg=reg, a_dt=_GP, c_dt=_GP, modifiers={_opd3_mod.SHL})

    def shift_greg_right(self, *, reg : greg_base, bit_count : int) -> str:
        return self.shift(adreg=reg, bimm=bit_count, cdreg=reg, a_dt=_GP, c_dt=_GP, modifiers={_opd3_mod.SHR})

    def zero_greg(self, *, greg : greg_base) -> str:
        return self.set(dreg=greg, dt=_ADT.UINT64, modifiers={_opset_mod.CONST}, constant="zero")

    def greg(self, reg_idx : int) -> greg_base:
        return aarch64_greg(reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        return aarch64_freg(reg_idx=reg_idx, dt=dt)
