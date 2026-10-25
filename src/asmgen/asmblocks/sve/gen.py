# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE asm generator
"""

from ..aarch64 import aarch64

from ...registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    ait_size,
    treg_base, vreg_base, freg_base, greg_base
)

from .types import sve_vreg,sve_preg
from .opd3 import sve_fma,sve_mul,sve_add
from .opmem.load import sve_load
from .opmem.store import sve_store

from .move import sve_move

from ..neon import neon
from ..aarch64.opd3 import aarch64_sub
from ..aarch64.opd3 import aarch64_shift
from ...registers import asm_data_type as _ADT
from .opset import sve_opset
from ..op.opset import opset_modifier as _opset_mod
from ..op.opd3 import opd3_modifier as _opd3_mod

class sve(aarch64):
    """
    SVE asmgen implementation
    """
    # FEAT_SVE2 (fmlalb/fmlalt), FEAT_BF16, FEAT_FP8FMA, FEAT_I8MM
    FEATURES = ('SVE2', 'BF16', 'B16B16', 'FP8FMA', 'I8MM')


    dt_suffixes = neon.dt_idxsuffixes

    dt_mnem_suffixes = {
            adt.FP128  : "q",
            adt.DOUBLE : "d",
            adt.SINT64 : "d",
            adt.UINT64 : "d",
            adt.SINGLE : "w",
            adt.SINT32 : "w",
            adt.UINT32 : "w",
            adt.HALF   : "h",
            adt.SINT16 : "h",
            adt.UINT16 : "h",
            adt.FP8E4M3: "b",
            adt.FP8E5M2: "b",
            adt.UINT8  : "b",
            adt.SINT8  : "b",
            }

    def isa_setup(self, predicate : int = 0):
        """
        What this ISA needs set up before a kernel and torn down after

        :param predicate: predicate register for the all-true predicate
        """
        from ..setup import sve_setup
        return sve_setup(self, predicate=predicate)

    def __init__(self):
        super().__init__()
        self.fma = sve_fma(asmwrap=self.asmwrap,
                           dt_suffixes=self.dt_suffixes,
                           dt_idxsuffixes=self.dt_suffixes)
        self.mul = sve_mul(asmwrap=self.asmwrap,
                           dt_suffixes=self.dt_suffixes,
                           dt_idxsuffixes=self.dt_suffixes)
        self.add = sve_add(asmwrap=self.asmwrap,
                             dt_suffixes=self.dt_suffixes,
                             dt_idxsuffixes=self.dt_suffixes)

        self.load = sve_load(asmwrap = self.asmwrap)
        self.store = sve_store(asmwrap = self.asmwrap)
        self.move = sve_move(asmwrap=self.asmwrap,
                             dt_suffixes=self.dt_suffixes)
        self.sub = aarch64_sub(asmwrap=self.asmwrap)
        self.shift = aarch64_shift(asmwrap=self.asmwrap)
        self.set = sve_opset(asmwrap=self.asmwrap, dt_suffixes=self.dt_suffixes)
        # the operations exist: the features that are off hide their signatures
        self._gate_features()

    def get_req_flags(self) -> list[str]:
        """
        Return required flags in cpuinfo for this generator to be supported
        """
        return ['sve']

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        req_flags = self.get_req_flags()
        supported = True
        for r in req_flags:
            if -1 == cpuinfo.find(r):
                supported = False
                break
        return supported

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        asmblock  = self.asmwrap(f"fcmne p1.d,{vreg1}.{suf},p0/z,#0,0")
        asmblock += self.asmwrap( "ptest p0, p1.b")
        asmblock += self.asmwrap(f"b.any .{label}")
        return asmblock


    @property
    def is_vla(self) -> bool:
        return True
    def max_vregs(self) -> int:
        return 32
    def max_mregs(self):
        return 8

    @property
    def simd_size(self) -> int:
        # compile-time knowledge: a vector-length-agnostic ISA's length is the
        # machine's, known at run time (set with VLEN, c_simd_size_function)
        raise NotImplementedError(f"{type(self).__name__} is vector-length-agnostic: its vector "
                                  "length is known at run time only (set with VLEN, "
                                  "c_simd_size_function)")


    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt) -> str:
        """
        A register moved by offset vector lengths in bytes: incb forward, decb
        back, at most 16 a time
        """
        # whole vector lengths: incb or decb, 16 at a time
        out, left = "", offset
        while left:
            n = max(-16, min(16, left))
            out += self.add(adreg=reg, bimm=n, cdreg=reg, a_dt=_ADT.UINT64, c_dt=_ADT.UINT64,
                            modifiers={_opd3_mod.VOFFSET})
            left -= n
        return out

    def mask_all_true(self, *, reg, dt : adt) -> str:
        return self.set(dreg=reg, dt=dt, modifiers={_opset_mod.CONSTMASK}, mask="alltrue")

    def zero_vreg(self, *, vreg : vreg_base, dt : adt) -> str:
        return self.set(dreg=vreg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")

    def vreg(self, reg_idx : int) -> sve_vreg:
        return sve_vreg(reg_idx)

    @property
    def max_add_voff(self) -> int:
        return 16

    def mreg(self, idx : int, is_pn : bool = False) -> sve_preg:
        """
        returns an SVE predicate register with the specified index

        :param idx: Register index
        :type idx: int
        :param is_pn: Is this a pn reg? (for SME/SME2)
        :type is_pn: bool
        :return: Predicate register
        :rtype: class:`asmgen.asmblocks.sve.types.sve_preg`
        """
        return sve_preg(idx, is_pn)

    def ptrue(self, reg : sve_preg, dt : adt) -> str:
        """
        Stores an all-true mask for the specified data type in the specified
        predicate register

        :param reg: Predicate register
        :type reg: class:`asmgen.asmblocks.sve.types.sve_preg`
        :param dt: Data type to use
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String with the required SVE ASM
        :rtype: str
        """
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ptrue {reg}.{suf}")


    def max_tregs(self, dt : adt) -> int:
        return 0

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError("SVE has no tiles, use SME")

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")
