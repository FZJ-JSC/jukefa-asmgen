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
from ..aarch64.opd3 import aarch64_shl, aarch64_shr

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
        self.shl = aarch64_shl(asmwrap=self.asmwrap)
        self.shr = aarch64_shr(asmwrap=self.asmwrap)

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
        return 1


    def simd_size_to_greg(self, *, reg: greg_base, dt: adt) -> str:
        suf = self.dt_mnem_suffixes[dt]
        result  = self.asmwrap(f"mov {reg},#0")
        result += self.asmwrap(f"inc{suf} {reg}")

        return result

    def indexable_elements(self, dt : adt) -> int:
        # 128 bits are indexable
        return 16//adt_size(dt)

    @property
    def c_simd_size_function(self) -> str:
        pre_oi = self.output_inline
        self.set_output_inline(yesno=True)
        result  = "inline size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap("mov %[byte_size],#0")
        result += "        "+self.asmwrap("incb %[byte_size]")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        self.set_output_inline(yesno=pre_oi)
        return result

    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt) -> str:
        """
        A register moved by offset vector lengths in bytes: incb forward, decb
        back, at most 16 a time
        """
        out, left = "", offset
        while left:
            n = max(-16, min(16, left))
            out += self.asmwrap(f"{'incb' if n > 0 else 'decb'} {reg}, ALL, MUL #{abs(n)}")
            left -= n
        return out

    def vlen_to_greg(self, *, reg, dt : adt) -> str:
        """
        The vector length in elements of dt into a general register: cnt{b,h,w,d}
        """
        suf = {1: 'b', 2: 'h', 4: 'w', 8: 'd'}[adt_size(dt)]
        return self.asmwrap(f"cnt{suf} {reg}")

    def mask_all_true(self, *, reg, dt : adt) -> str:
        return self.ptrue(reg, dt)

    def greg_to_voffs(self, *, streg : greg_base, vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        cstreg = str(streg)
        if adt_size(dt) < 8:
            cstreg = streg.get_wreg()
        return self.asmwrap(f"index {vreg}.{suf},#0,{cstreg}")


    def zero_vreg(self, *, vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"dup {vreg}.{suf},#0")

    def vreg(self, reg_idx : int) -> sve_vreg:
        return sve_vreg(reg_idx)

    def min_load_immoff(self, dt : adt) -> int:
        return 0

    def max_load_immoff(self, dt : adt) -> int:
        return 252

    def min_bcast_immoff(self, dt : adt) -> int:
        return self.min_load_immoff(dt)

    def max_bcast_immoff(self, dt : adt) -> int:
        return self.max_load_immoff(dt)

    @property
    def min_load_voff(self) -> int:
        return -8

    @property
    def max_load_voff(self) -> int:
        return 7

    @property
    def max_add_voff(self) -> int:
        return 16

    def fill_vector(self, *, sreg : freg_base,
                    vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"dup {vreg}.{suf}, z{sreg.idx}.{suf}[0]")

    def load_vector(self, *, areg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1{msuf} {vreg}.{suf}, p0/z, [{areg}]")

    def load_vector_voff(self, *, areg : greg_base, voffset : int,
                         vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        if voffset > self.max_load_voff:
            raise ValueError(f"voffset {voffset} > {self.max_load_voff}")
        return self.asmwrap(f"ld1{msuf} {vreg}.{suf}, p0/z, [{areg}, #{voffset}, MUL VL]")

    def load_vector_immoff(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
            "SVE has no vector loads with immediate offset, use load_vector_voff")

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1r{msuf} {vreg}.{suf}, p0/z, [{areg}]")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1r{msuf} {vreg}.{suf}, p0/z, [{areg}, #{offset}]")

    def load_vector_bcast1_inc(self, *, areg : greg_base, offset : int,
                              vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
                "SVE doesn't have a post-index ld1r{suf}, use load_vector_bcast1_immoff instead")

    def store_vector_voff(self, *, areg : greg_base, voffset : int,
                          vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        address = f"[{areg}, #{voffset}, MUL VL]"
        if 0 == voffset:
            address = f"[{areg}]"
        return self.asmwrap(f"st1{msuf} {{{vreg}.{suf}}}, p0, {address}")

    def store_vector_immoff(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
            "SVE has no vector stores with immediate offset, use store_vector_voff")

    def store_vector(self, *, areg : greg_base,
                     vreg : vreg_base, dt : adt) -> str:
        return self.store_vector_voff(areg=areg, voffset=0, vreg=vreg, dt=dt)

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


    def load_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no load with immediate stride")

    def load_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no load with scalar register stride")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait) -> str:
        if ait_size(it) not in [4,8]:
            raise ValueError("index size must be 4 or 8 bytes!")
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        xtnd = ""
        if (ait_size(it) == 4) and (adt_size(dt) == 4):
            xtnd = ", SXTW"
        return self.asmwrap(f"ld1{msuf} {vreg}.{suf}, p0/z,[{areg}, {offvreg}.{suf}{xtnd}]")

    def store_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("RVV has no store with immediate stride")

    def store_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no store with scalar register stride")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt,
                             it : ait) -> str:
        if ait_size(it) not in [4,8]:
            raise ValueError("index size must be 4 or 8 bytes!")
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        xtnd = ""
        if (ait_size(it) == 4) and (adt_size(dt) == 4):
            xtnd = ", SXTW"
        return self.asmwrap(f"st1{msuf} {vreg}.{suf}, p0, [{areg}, {offvreg}.{suf}{xtnd}]")


    # Unsupported functionality:
    def max_tregs(self, dt : adt) -> int:
        return 0

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError("SVE has no tiles, use SME")

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")

    def load_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        raise NotImplementedError("Lane loads not yet implemented for SVE")

    def store_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        raise NotImplementedError("Lane stores not yet implemented for SVE")
