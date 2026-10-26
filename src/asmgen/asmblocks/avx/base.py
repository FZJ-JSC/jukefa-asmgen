# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Shared AVX(X86_64) generator base class (avxbase)
"""

from copy import deepcopy
from typing import Union
from abc import abstractmethod

from ..noarch import compiler_family
from ..x86.opmem.displacement import ENCODING_FEATURES as X86_ENCODING_FEATURES
from ...asmdata import asm_data
from ...registers import asm_index_type
from ...registers import (
    adt_size,
    freg_base,

    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base, vreg_base, freg_base, greg_base
)

from ..noarch import asmgen,comparison
from ...callconv.callconv import callconv

from .types import x86_greg,avx_freg,xmm_vreg,ymm_vreg,zmm_vreg,reg_prefixer,avx512_mreg

from .opd3 import avx_fma,avx_mul,avx_add

from .opmem.load import avx128_load,avx256_load,avx512_load
from .opmem.store import avx128_store,avx256_store,avx512_store
from ...registers import asm_data_type as _ADT
from ..op.operand import operand_modifier as _opd_mod
from ..op.move import move_modifier as _move_mod
from ..op.scalar import BITS_GP as _BITS_GP
from ..op.opset import opset_modifier as _opset_mod
from ..x86.types import DT_SUFFIXES
from ..op.opd3 import opd3_modifier as _opd3_mod
from ..x86.types import SIZE_SUFFIXES

_GP = _ADT.UINT64  # general registers' arithmetic: 64-bit

class avxbase(asmgen):
    """
    Base X86_64/AVX/FMA asmgem implementation
    """

    #: the nop: nop
    NOP_BYTES = 1
    NOP_ENCODING = 0x90

    def compiler_flags(self, compiler : str = "gcc") -> list[str]:
        if "arm" == compiler_family(compiler):
            raise NotImplementedError("armclang targets no x86")
        return ["-mavx2", "-mfma"]

    #: x86's displacements: disp8 on, disp32 off by default (x86/opmem/displacement.py)
    ENCODING_FEATURES = dict(X86_ENCODING_FEATURES)

    greg_names = [f'r{i}' for i in \
            [str(j) for j in range(8,16)]+\
            ['ax','bx','cx','dx','si','di','bp','sp']]

    cb_insts = {
            'NZ' : 'jnz',
            'EZ' : 'jz',
            'NE' : 'jne',
            'EQ' : 'je',
            'LE' : 'jle',
            'GE' : 'jge',
            'LT' : 'jl',
            'GT' : 'jg',
            }

    def isa_setup(self):
        """
        What this ISA needs set up before a kernel and torn down after
        """
        from ..setup import constants_setup
        return constants_setup(self)

    def __init__(self):
        super().__init__()
        self.callconvs = {
            "systemv" : callconv(
                param_regs={ 'greg' : [13,12,11,10, 0,1], # RDI, RSI, RDX, RCX, R8, R9
                             'freg' : list(range(0, 8))},
                caller_save_lists={ 'greg' : [13,12,11,10, 0,1],  # func args/return vals
                                    'freg' : list(range(0, 8))
                                   },
                callee_save_lists={ 'greg' : list(range(4,8)) +# R12-R15
                                        [9,   # BX
                                         14,  # Frame Pointer
                                         15], # SP
                                    # SysV: every xmm register is caller-saved
                                    'freg' : []},
                spreg=15,
                spstart=8)
            }
        self.default_callconv = "systemv"

        self.rpref = reg_prefixer(
                output_inline_getter=lambda : self.output_inline)

    def create_callconv(self, name : str = "default"):

        if "default" == name:
            name = self.default_callconv

        return deepcopy(self.callconvs[name])

    @abstractmethod
    def get_req_flags(self) -> list[str]:
        """
        Return required flags in cpuinfo for this generator to be supported
        """
        raise NotImplementedError("Not implemented for base class")

    def supportedby_cpuinfo(self, cpuinfo : str):
        req_flags = self.get_req_flags()
        supported = True
        for r in req_flags:
            if -1 == cpuinfo.find(r):
                supported = False
                break
        return supported

    dt_suffixes = DT_SUFFIXES
    size_suffixes = SIZE_SUFFIXES

    size_mask_suffixes = {
            1 : 'q', # AVX512BW
            2 : 'd', # AVX512BW
            4 : 'w', # AVX512F
            8 : 'b', # AVX512DQ
            }

    it_suffixes = {
            ait.INT64 : "q",
            ait.INT32 : "d",
            }

    def shared_registers(self) -> list[tuple[str, str, dict[int, int] | None]]:
        # a scalar FP register is the low part of the vector register
        return [("freg", "vreg", None), ("mreg", "vreg", None)]

    def reserved_registers(self) -> dict[str, dict[str, int]]:
        # indices as x86_greg names them: rbp, rsp; k0 can't be a write mask
        return {"greg": {"rbp": 14, "rsp": 15}}

    def iota_label(self, size : int, count : int) -> str:
        """
        Returns the label string for iota data (used for emulating strided ld/st
        with gather/scatter)

        :param size: size of the data in bytes
        :type size: int
        :param count: Number of elements
        :type count: int
        :return: string to be used for the label
        :rtype: str
        """
        return f"iota_{size*8}x{count}"

    def index_table(self, dt : adt) -> str:
        """
        The table 0, 1, 2, ... of dt's element size for a vector, registered in
        the generator's data where it isn't yet: its label, as written
        """
        count = self.simd_size // adt_size(dt)
        self.ensure_indices(dt, count)
        return self.labelstr(self.iota_label(adt_size(dt), count))

    def ensure_indices(self, dt : adt, count : int):
        """
        Ensures iota data for this data type and element count exists

        :param dt: element data type
        :type dt: class:`asmgen.registers.asm_data_type`
        :param count: Number of elements
        :type count: int
        """
        dt_size = adt_size(dt)
        key = self.iota_label(dt_size, count)
        if key in self.asmdata:
            return

        #TODO: Might want to rework this, maybe ait param and
        #      then introduce ait_to_adt or something?
        index_dt : adt = None
        if dt_size == 1:
            index_dt = adt.SINT8
        elif dt_size == 2:
            index_dt = adt.SINT16
        elif dt_size == 4:
            index_dt = adt.SINT32
        elif dt_size == 8:
            index_dt = adt.SINT64
        else:
            raise ValueError(f"Can't determine index type for {dt}")


        self.asmdata[key] = [asm_data(index_dt, i) for i in range(count)]


    def jzero(self, *, reg : greg_base, label : str) -> str:
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz {self.labelstr(label)}")
        return asmblock

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        suf = 's'+self.dt_suffixes[dt]
        pfreg1 = self.rpref(freg1)
        pfreg2 = self.rpref(freg2)
        asmblock  = self.set(dreg=freg2, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")
        asmblock += self.asmwrap(f"ucomi{suf} {pfreg2},{pfreg1}")
        asmblock += self.asmwrap(f"je {self.labelstr(label)}")
        return asmblock

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base, greg : greg_base, label : str,
               dt : adt) -> str:
        suf = 'p'+self.dt_suffixes[dt]
        pvreg1 = self.rpref(vreg1)
        pvreg2 = self.rpref(vreg2)
        asmblock  = self.set(dreg=vreg2, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")
        asmblock += self.asmwrap(f"vcmpeq{suf} {pvreg2},{pvreg1},{pvreg2}")
        asmblock += self.asmwrap(f"vptest {pvreg2},{pvreg2}")
        asmblock += self.asmwrap(f"jne {self.labelstr(label)}")
        return asmblock

    def fcb(self, *, reg1, reg2, cmp: comparison, label: str, dt: adt, greg=None) -> str:
        """
        cb on FP registers: equality as floating point, so -0.0 equals 0.0
        and NaN equals nothing
        """
        if reg2 is None:
            raise ValueError("x86 compares FP registers with another: pass reg2")
        suf = 'd' if 8 == adt_size(dt) else 's'
        lab = self.labelstr(label)
        out = self.asmwrap(f"vucomis{suf} {self.rpref(reg2)},{self.rpref(reg1)}")
        # unordered (NaN) sets the parity flag: not equal
        if cmp.name in ('NE', 'NZ'):
            return out + self.asmwrap(f"jne {lab}") + self.asmwrap(f"jp {lab}")
        if cmp.name in ('EQ', 'EZ'):
            return out + self.asmwrap("jp 1f") + self.asmwrap(f"je {lab}") + self.asmwrap("1:")
        raise NotImplementedError(f"x86 FP comparison {cmp.name}")

    def cb(self, *, reg1, reg2, cmp: comparison, label: str,
           dt: adt | None = None, greg: greg_base | None = None) -> str:
        if isinstance(reg1, freg_base):
            return self.fcb(reg1=reg1, reg2=reg2, cmp=cmp, label=label, dt=dt, greg=greg)
        inst = self.cb_insts[cmp.name]
        if reg2 is None:
            pr1 = self.rpref(reg1)
            return self.asmwrap(f"test {pr1},{pr1}")+\
                   self.asmwrap(f"{inst} {self.labelstr(label)}")

        pr1 = self.rpref(reg1)
        pr2 = self.rpref(reg2)
        return self.asmwrap(f"cmp {pr2},{pr1}")+\
               self.asmwrap(f"{inst} {self.labelstr(label)}")

    def xmm_to_ymm(self, vreg : vreg_base):
        """
        Returns the expanded ymm register from the specified xmm register
        
        :param vreg: XMM register to convert to YMM
        :type vreg: class:`asmgen.asmblocks.avx.types.xmm_vreg`
        :return: Corresponding YMM register
        :rtype vreg: class:`asmgen.asmblocks.avx.types.ymm_vreg`
        """
        if not isinstance(vreg, xmm_vreg):
            raise ValueError(f"{vreg} is not an XMM register")

        idx = 0
        xmm_str = f"{vreg}"
        if xmm_str.startswith("xmm"):
            idx = int(xmm_str[3:])
        elif xmm_str.startswith("%xmm"):
            idx = int(xmm_str[4:])
        elif xmm_str.startswith("%%xmm"):
            idx = int(xmm_str[5:])
        else:
            raise ValueError("Not an XMM register: {vreg}")
        return ymm_vreg(idx)

    def label(self, *, label : str) -> str:
        asmblock = self.asmwrap(f"{self.labelstr(label)}:")
        return asmblock

    def jump(self, *, label : str) -> str:
        asmblock = self.asmwrap(f"jmp {self.labelstr(label)}")
        return asmblock

    def loopbegin(self, *, reg : greg_base, label : str):
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopbegin_nz(self, *, reg : greg_base, label : str, labelskip : str):
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz {self.labelstr(labelskip)}")
        asmblock += self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopend(self, *, reg : greg_base, label : str):
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"cmp $0x0,{preg}")
        asmblock += self.asmwrap(f"jne {self.labelstr(label)}")

        return asmblock

    @property
    def is_vla(self):
        return False

    def max_gregs(self):
        return 16
    def max_mregs(self):
        # every vector register can be a mask (shared_registers)
        return self.max_vregs()

    def kiterkleft_pow2(self, *, kreg : greg_base,
                        kleftreg : greg_base,
                        unroll : int) -> str:
        """
        kiterkleft variant for unroll values that are powers of 2
        
        Can be simplified down to a "right shift" and "and"
        """

        asmblock = ""

        unrollpower = unroll.bit_length()-1

        pkreg = self.rpref(kreg)
        pkleftreg = self.rpref(kleftreg)

        asmblock += self.move(dregs=[kreg, kleftreg], dts=[_GP, _GP])
        # Example unroll=4: kleft = k & 0b0111
        asmblock += self.asmwrap(f"andq ${unroll-1}, {pkleftreg}")
        # Example unroll=4: k = k >> 2
        asmblock += self.asmwrap(f"shrq ${unrollpower}, {pkreg}")

        return asmblock

    # pylint: disable-next=too-many-branches
    def kiterkleft(self, *, kreg : greg_base,
                   kleftreg : greg_base,
                   tmpreg : greg_base,
                   unroll : int) -> str:

        if unroll.bit_count() == 1:
            return self.kiterkleft_pow2(kreg=kreg,kleftreg=kleftreg,unroll=unroll)

        #TODO: division by invariant multiplication

        # TODO: figure out some elaborate way to deal with implicitly used registers
        #       (maybe also relevant: register restriction for col/row in ARM SME)
        asmblock = ""
        rax = self.rpref(self.greg(8))
        rdx = self.rpref(self.greg(11))
        ptmpreg = self.rpref(tmpreg)
        pkreg = self.rpref(kreg)
        pkleftreg = self.rpref(kleftreg)

        # 1) Put k into RAX and clear RDX
        # 2) Put unroll into some register (not RAX or RDX)
        # 3) div
        # 4) put RAX into kreg
        # 5) put RDX into kleftreg


        # save rax/rdx unless they're supposed to get overwritten anyways
        if f"{rax}" not in [f"{pkreg}", f"{pkleftreg}", f"{ptmpreg}"]:
            asmblock += self.asmwrap(f"push {rax}")
        if f"{rdx}" not in [f"{pkreg}", f"{pkleftreg}", f"{ptmpreg}"]:
            asmblock += self.asmwrap(f"push {rdx}")

        if f"{pkreg}" != f"{rax}":
            asmblock += self.move(dregs=[kreg, self.greg(8)], dts=[_GP, _GP]) #RAX
        if f"{pkleftreg}" not in [f"{rax}",f"{rdx}"]:
            unrollreg = kleftreg
        elif f"{pkreg}" not in [f"{rax}",f"{rdx}"]:
            unrollreg = kreg
        else:
            unrollreg = tmpreg

        punrollreg = self.rpref(unrollreg)

        asmblock += self.asmwrap(f"xorq {rdx},{rdx}")
        asmblock += self.set(dreg=unrollreg, dt=_GP, modifiers={_opset_mod.CONST}, value=unroll)

        # The DIV
        asmblock += self.asmwrap(f"div {punrollreg}")


        if f"{pkreg}" != f"{rdx}":
            if f"{pkreg}" != f"{rax}":
                asmblock += self.move(dregs=[self.greg(8), kreg], dts=[_GP, _GP]) #RAX
            if f"{pkleftreg}" != f"{rdx}":
                asmblock += self.move(dregs=[self.greg(11), kleftreg], dts=[_GP, _GP]) #RDX
        elif f"{pkleftreg}" != f"{rax}":
            if f"{pkleftreg}" != f"{rdx}":
                asmblock += self.move(dregs=[self.greg(11), kleftreg], dts=[_GP, _GP]) #RDX
            if f"{pkreg}" != f"{rax}":
                asmblock += self.move(dregs=[self.greg(8), kreg], dts=[_GP, _GP]) #RAX
        else:
            # exchange rax and rdx
            asmblock += self.asmwrap(f"xorq {rax},{rdx}")
            asmblock += self.asmwrap(f"xorq {rdx},{rax}")
            asmblock += self.asmwrap(f"xorq {rax},{rdx}")

        # restore rax/rdx unless they're supposed to get overwritten anyways
        if f"{rdx}" not in [f"{pkreg}", f"{pkleftreg}", f"{ptmpreg}"]:
            asmblock += self.asmwrap(f"pop {rdx}")
        if f"{rax}" not in [f"{pkreg}", f"{pkleftreg}", f"{ptmpreg}"]:
            asmblock += self.asmwrap(f"pop {rax}")

        return asmblock

    def greg(self, reg_idx : int) -> greg_base:
        return x86_greg(reg_idx=reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        _ = dt # explicitly unused
        return avx_freg(reg_idx=reg_idx)

    def mreg(self, reg_idx : int):
        """
        AVX2's masks are vector registers - gathers', masked loads' and stores',
        variable blends' - any of them: mask i is vector register i
        """
        return self.vreg(reg_idx)

    def max_tregs(self, dt : adt):
        return 0


    avx_no_tile_message = "Not tile support in AVX (will be supported with AMX)"

    def treg(self, reg_idx : int, dt : adt):
        raise NotImplementedError(self.avx_no_tile_message)
