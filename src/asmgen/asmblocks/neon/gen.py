# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD asm generator and related types
"""
from ...registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base,vreg_base,freg_base,greg_base
)
from ..aarch64 import aarch64

from ..aarch64.types import aarch64_freg
from .types import neon_vreg
from .opd3 import neon_fma,neon_mul,neon_add
from .opmem import neon_load, neon_store
from .move import neon_move
from ..aarch64.opd3 import aarch64_sub
from ..aarch64.opd3 import aarch64_shl, aarch64_shr
from ...registers import asm_data_type as _ADT

class neon(aarch64):
    """
    NEON/ASIMD implementation of asmgen
    """
    # FEAT_FP16, FEAT_FHM (fmlal), FEAT_BF16, FEAT_FP8FMA, FEAT_DotProd, FEAT_I8MM
    FEATURES = ('FP16', 'FHM', 'BF16', 'FP8FMA', 'DOTPROD', 'I8MM')


    dt_suffixes = {
            adt.DOUBLE  : "2d",
            adt.UINT64  : "2d",
            adt.SINT64  : "2d",
            adt.SINGLE  : "4s",
            adt.UINT32  : "4s",
            adt.SINT32  : "4s",
            adt.HALF    : "8h",
            adt.UINT16  : "8h",
            adt.SINT16  : "8h",
            adt.FP8E5M2 : "16b",
            adt.FP8E4M3 : "16b",
            adt.UINT8   : "16b",
            adt.SINT8   : "16b",
            }
    dt_idxsuffixes = {
            adt.DOUBLE  : "d",
            adt.UINT64  : "d",
            adt.SINT64  : "d",
            adt.SINGLE  : "s",
            adt.UINT32  : "s",
            adt.SINT32  : "s",
            adt.HALF    : "h",
            adt.UINT16  : "h",
            adt.SINT16  : "h",
            adt.FP8E5M2 : "b",
            adt.FP8E4M3 : "b",
            adt.UINT8   : "b",
            adt.SINT8   : "b",
            }

    dt_greg_pfx = {
            adt.DOUBLE  : "x",
            adt.SINGLE  : "w",
            adt.HALF    : "w",
            adt.FP8E5M2 : "w",
            adt.FP8E4M3 : "w",
            }

    def __init__(self):
        super().__init__()
        self.fma = neon_fma(asmwrap=self.asmwrap,
                            dt_suffixes=self.dt_suffixes,
                            dt_idxsuffixes=self.dt_idxsuffixes)
        self.mul = neon_mul(asmwrap=self.asmwrap,
                              dt_suffixes=self.dt_suffixes,
                              dt_idxsuffixes=self.dt_idxsuffixes)
        self.add = neon_add(asmwrap=self.asmwrap,
                              dt_suffixes=self.dt_suffixes,
                              dt_idxsuffixes=self.dt_idxsuffixes)

        self.load = neon_load(asmwrap=self.asmwrap)
        self.store = neon_store(asmwrap=self.asmwrap)
        self.move = neon_move(asmwrap=self.asmwrap, dt_suffixes=self.dt_suffixes,
                              dt_idxsuffixes=self.dt_idxsuffixes)
        self.sub = aarch64_sub(asmwrap=self.asmwrap)
        self.shl = aarch64_shl(asmwrap=self.asmwrap)
        self.shr = aarch64_shr(asmwrap=self.asmwrap)

    def get_req_flags(self) -> list[str]:
        """
        Return required flags in cpuinfo for this generator to be supported
        """
        return ['asimd']

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
        asmblock  = self.asmwrap(f"fmaxv {freg}, {vreg2}.{suf}")
        asmblock += self.asmwrap(f"fcmp {freg}, #0.0")
        asmblock += self.asmwrap(f"b.eq .{label}")
        return asmblock

    def vreg_to_qreg(self, vreg : neon_vreg) -> aarch64_freg:
        """
        Returns the AArch64 128 FP register register corresponding to
        the specified vector register

        :param vreg: NEON/ASIMD vector register
        :type vreg: class:`asmgen.asmblocks.neon.types.neon_vreg`
        :return: 128 bit FP reg
        :rtype: class:`asmgen.asmblocks.aarch64.types.aarch64_freg`
        """
        return aarch64_freg(reg_idx=vreg.idx, dt=adt.FP128)

    @property
    def is_vla(self) -> bool:
        return False

    def indexable_elements(self, dt : adt) -> int:
        return self.simd_size//adt_size(dt)
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 16

    def simd_size_to_greg(self, *, reg: greg_base, dt: adt) -> str:
        return self.mov_greg_imm(reg=reg, imm=16//adt_size(dt))

    @property
    def c_simd_size_function(self):
        return f"inline size_t get_simd_size() {{ return {self.simd_size}; }}"

    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt):
        # a fixed vector length: the byte count, within add's immediates
        return self.add(adreg=reg, imm=self.simd_size*offset, cdreg=reg, a_dt=_ADT.UINT64, c_dt=_ADT.UINT64)

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        suf = self.dt_suffixes[dt]
        zeroreg = f"{self.dt_greg_pfx[dt]}zr"
        return self.asmwrap(f"dup {vreg}.{suf},{zeroreg}")

    def vreg(self, reg_idx : int) -> neon_vreg:
        return neon_vreg(reg_idx)

    def qreg(self, idx : int) -> aarch64_freg:
        """
        Returns the AArch64 128 FP register register corresponding to
        the specified register id
        :param idx: register id
        :type idx: int
        :return: 128 bit FP reg
        :rtype: class:`asmgen.asmblocks.aarch64.types.aarch64_freg`
        """
        return aarch64_freg(reg_idx=idx, dt=adt.FP128)

    def min_load_immoff(self, dt : adt) -> int:
        _ = dt # explicitly unused
        return 0

    def max_load_immoff(self, dt : adt) -> int:
        return 4095*adt_size(dt)*2

    def min_bcast_immoff(self, dt : adt) -> int:
        _ = dt # explicitly unused
        return 0

    def max_bcast_immoff(self, dt : adt) -> int:
        _ = dt # explicitly unused
        return 0

    @property
    def min_load_voff(self) -> int:
        return 0

    @property
    def max_load_voff(self) -> int:
        return 4096//8

    @property
    def max_add_voff(self) -> int:
        return 4096//self.simd_size

    def greg_to_voffs(self, *, streg : greg_base, vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
                "NEON has no instruction for creating vector indices")

    def fill_vector(self, *, sreg : freg_base,
                    vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_suffixes[dt]
        isuf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"dup {vreg}.{suf}, v{sreg.idx}.{isuf}[0]")

    def load_vector(self, *, areg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}]")

    def load_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"ld1 {{{vreg}.{suf}}}[{lane}], [{areg}]")

    def load_vector_lane_greginc(self, *, areg : greg_base, offreg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        """
        load_vector_lane variant with post-index increment by
        a value in a GP register
        """
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"ld1 {{{vreg}.{suf}}}[{lane}], [{areg}], {offreg}")

    def load_vector_lane_inc(self, *, areg : greg_base, offset : int,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        """
        load_vector_lane variant with post-index increment by
        an immediate
        """
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"ld1 {{{vreg}.{suf}}}[{lane}], [{areg}], #{offset}")

    def load_vector_voff(self, *, areg : greg_base, voffset : int,
                         vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}, #{voffset*self.simd_size}]")

    def load_vector_immoff(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}, #{offset}]")

    def load_vector_inc(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        """
        load_vector variant with post-index increment by
        an immediate
        """
        _ = dt # explicitly unused
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}], #{offset}")

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ld1r {{{vreg}.{suf}}}, [{areg}]")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
                ("load_vector_bcast1_immoff doesn't make sense with NEON,"
                 " use load_vector_bcast1_inc or load_vector_voff + fma_idx instead"))

    def load_vector_bcast1_inc(self, *, areg : greg_base, offset : int,
                              vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ld1r {{{vreg}.{suf}}}, [{areg}], #{offset}")

    def store_vector_voff(self, *, areg : greg_base, voffset : int,
                          vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}, #{voffset*self.simd_size}]")

    def store_vector_immoff(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        _ = dt # explicitly unused
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}, #{offset}]")

    def store_vector_inc(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        """
        store_vector variant with post-index increment by
        an immediate
        """
        _ = dt # explicitly unused
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}], #{offset}")

    def store_vector(self, *, areg : greg_base,
                     vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}]")

    def store_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"st1 {{{vreg}.{suf}}}[{lane}], [{areg}]")

    def store_vector_lane_greginc(self, *, areg : greg_base, offreg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        """
        store_vector_lane variant with post-index increment by
        a value in a GP register
        """
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"st1 {{{vreg}.{suf}}}[{lane}], [{areg}], {offreg}")

    def store_vector_lane_inc(self, *, areg : greg_base, offset : int,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        """
        store_vector_lane variant with post-index increment by
        an immediate
        """
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        suf = self.dt_idxsuffixes[dt]
        return self.asmwrap(f"st1 {{{vreg}.{suf}}}[{lane}], [{areg}], #{offset}")

    def load_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no load with immediate stride")

    def load_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no load with scalar register stride")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait) -> str:
        raise NotImplementedError("NEON has no load with vector register stride")

    def store_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no store with immediate stride")

    def store_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no store with scalar register stride")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt,
                             it : ait) -> str:
        raise NotImplementedError("NEON has no store with vector register stride")

    # Unsupported functionality:
    def max_tregs(self, dt : adt) -> int:
        return 0

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError("NEON has no tiles, use SME")

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no tiles, use SME")

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("NEON has no tiles, use SME")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("NEON has no tiles, use SME")
