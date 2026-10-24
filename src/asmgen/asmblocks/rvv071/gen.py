# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V RVV 0.7.1 asm generator and related types
"""

from ..rvv import rvv

from ...registers import asm_data_type as adt, adt_size, greg_base, vreg_base
from ...registers import asm_index_type as ait
from ..rvv.opset import rvv_opset

# pylint: disable=too-many-public-methods
class rvv071(rvv):
    """
    RISC-V RVV 0.7.1 asmgen implementation
    """

    def __init__(self):
        super().__init__()
        # vsetvli without tail and mask policies
        self.set = rvv_opset(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul, policy=False)

    dt_suffixes = {
            adt.DOUBLE  : "e",
            adt.SINGLE  : "w",
            adt.HALF    : "h",
            adt.FP8E4M3 : "b",
            adt.FP8E5M2 : "b",
            }

    def vsetvli(self, *, vlreg : greg_base, avlreg : greg_base, dt : adt) -> str:
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {vlreg}, {avlreg}, {dt_size}, m1")

    def vsetvlmax(self, *, reg : greg_base, dt : adt) -> str:
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {reg}, zero, {dt_size}, m1")

    # RVV's old vector loads and stores, which print 0.7.1's mnemonics through
    # dt_suffixes: RVV itself has them on load and store (0.6, phase 5), rvv071
    # keeps these until its port

    def load_vector(self, *, areg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vl{dt_suf}.v {vreg}, ({areg})")

    # I'm not seeing equivalents in RVV, I think you're supposed to do things differently
    # (LMUL > 1?), vector index?

    def load_vector_voff(self, *, areg : greg_base, voffset : int,
                         vreg : vreg_base, dt : adt) -> str:
        #raise NotImplementedError("RVV has no vector loads with address offset")
        # We can still load the vector - with max_load_{imm,v}off being 0, the generator will
        # just always pass an offset of 0 and add any offset to the address register after
        if voffset != 0:
            raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector(areg=areg, vreg=vreg, dt=dt)

    def load_vector_immoff(self, *, areg : greg_base, offset : int,
                         vreg : vreg_base, dt : adt) -> str:
        #raise NotImplementedError("RVV has no vector loads with address offset")
        # We can still load the vector - with max_load_{imm,v}off being 0, the generator will
        # just always pass an offset of 0 and add any offset to the address register after
        if offset != 0:
            raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector(areg=areg, vreg=vreg, dt=dt)

    def load_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        raise NotImplementedError("Lane loads not implemented for RVV")

    def load_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("RVV has no load with immediate stride")

    def load_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vls{dt_suf}.v {vreg}, ({areg}), {sreg}")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait) -> str:
        i_suf = self.it_suffixes[it]
        return self.asmwrap(f"vlux{i_suf}.v {vreg}, ({areg}), {offvreg}")

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt) -> str:
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vls{dt_suf}.v {vreg}, ({areg}), zero")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt) -> str:
        if offset != 0:
            raise NotImplementedError("RVV has no vector loads with address offset")
        return self.load_vector_bcast1(areg=areg, vreg=vreg, dt=dt)

    def load_vector_bcast1_inc(self, *, areg : greg_base, offset : int,
                              vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("RVV has no vector loads with address increment")

    def store_vector(self, *, areg : greg_base,
                     vreg : vreg_base, dt : adt) -> str:
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vs{dt_suf}.v {vreg}, ({areg})")

    def store_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        raise NotImplementedError("Lane stores not implemented for RVV")

    def store_vector_voff(self, *, areg : greg_base, voffset : int,
                          vreg : vreg_base, dt : adt) -> str:
        if voffset != 0:
            raise NotImplementedError("RVV has no vector stores with address offset")
        return self.store_vector(areg=areg, vreg=vreg, dt=dt)

    def store_vector_immoff(self, *, areg : greg_base, offset : int,
                         vreg : vreg_base, dt : adt) -> str:
        #raise NotImplementedError("RVV has no vector stores with address offset")
        # We can still store the vector - with max_store_{imm,v}off being 0, the generator will
        # just always pass an offset of 0 and add any offset to the address register after
        if offset != 0:
            raise NotImplementedError("RVV has no vector stores with address offset")
        return self.store_vector(areg=areg, vreg=vreg, dt=dt)

    def store_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("RVV has no store with immediate stride")

    def store_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        dt_suf = self.dt_suffixes[dt]
        return self.asmwrap(f"vss{dt_suf}.v {vreg}, ({areg}), {sreg}")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt,
                             it : ait) -> str:
        i_suf = self.it_suffixes[it]
        return self.asmwrap(f"vsux{i_suf}.v {vreg}, ({areg}), {offvreg}")

    # Unsupported functionality:
