# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V RVV 1.0 asm generator and related types
"""

from ...registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base,
    vreg_base, freg_base, greg_base
)
from ..riscv64 import riscv64
from .types import rvv_vreg

from .opd3 import rvv_fma,rvv_mul,rvv_add
from .opmem import rvv_load,rvv_store
from .move import rvv_move
from ..riscv64.opd3 import riscv64_sub
from ..riscv64.opd3 import riscv64_shl, riscv64_shr


# pylint: disable=too-many-public-methods

class rvv(riscv64):
    """
    RISC-V RVV 1.0 asmgen implementation
    """
    # Zvfh: half-precision vector arithmetic
    FEATURES = ('ZVFH', 'ZBA')


    dt_suffixes = {
            adt.DOUBLE  : "e64",
            adt.SINGLE  : "e32",
            adt.HALF    : "e16",
            adt.FP8E4M3 : "e8",
            adt.FP8E5M2 : "e8",
            }
    it_suffixes = {
            ait.INT64 : "ei64",
            ait.INT32 : "ei32",
            ait.INT16 : "ei16",
            ait.INT8  : "ei8",
            }

    def isa_setup(self):
        """
        What this ISA needs set up before a kernel and torn down after
        """
        from ..setup import rvv_setup
        return rvv_setup(self)

    def __init__(self):
        super().__init__()
        self.fma = rvv_fma(asmwrap=self.asmwrap)
        self.mul = rvv_mul(asmwrap=self.asmwrap)
        self.add = rvv_add(asmwrap=self.asmwrap)

        self.lmul = 1

        self.load = rvv_load(asmwrap=self.asmwrap,
                             lmul_getter=lambda : self.lmul)
        self.store = rvv_store(asmwrap=self.asmwrap,
                               lmul_getter=lambda : self.lmul)
        self.move = rvv_move(asmwrap=self.asmwrap)
        self.sub = riscv64_sub(asmwrap=self.asmwrap)
        self.shl = riscv64_shl(asmwrap=self.asmwrap)
        self.shr = riscv64_shr(asmwrap=self.asmwrap)

    def get_parameters(self) -> list[str]:
        return ["LMUL"]

    def get_param_value(self, name : str) -> int|str:
        if "LMUL" == name:
            return self.lmul
        return super().get_param_value(name)

    def set_parameter(self, name : str, value : int|str):
        if "LMUL" == name:
            if isinstance(value, str) and value.isdigit():
                self.lmul = int(value)
                if self.lmul.bit_count() != 1:
                    raise ValueError(f"Invalid LMUL {value}")
            elif isinstance(value, int):
                self.lmul = value
            else:
                raise NotImplementedError(
                        ("{value} is not an integer. Fractional "
                         "LMUL is not implemented yet"))
        else:
            raise ValueError(f"Invalid name {name} or value {value}")

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        isa_idx = cpuinfo.find("rv64")
        if -1 == isa_idx:
            return False
        isa_idx = isa_idx+4
        extensions = cpuinfo[isa_idx:].split()[0]
        print(f"Extensions: {extensions}")
        return "v" in extensions

    def vreg(self, reg_idx : int) -> vreg_base:
        return rvv_vreg(reg_idx * self.lmul)

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base, greg : greg_base, label : str,
               dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        asmblock  = self.asmwrap(f"fmv.{dt_suf}.x {freg},zero")
        # vec filled with 1 where element not-zero
        asmblock += self.asmwrap(f"vmfne.vf {vreg2},{vreg1},{freg}")
        # greg has number of elements that are not zero
        asmblock += self.asmwrap(f"vcpop.m {greg},{vreg2}")
        # if non-zero number of elements are non-zero, i.e greg has non-zero number,
        # we do not have zero, so we don't jump
        # So we jump when zero
        asmblock += self.asmwrap(f"beqz {greg},{label}")
        return asmblock

    @property
    def is_vla(self):
        return True

    def indexable_elements(self, dt : adt):
        return self.simd_size//adt_size(dt)
    def max_vregs(self):
        return 32//self.lmul

    @property
    def simd_size(self):
        return 1

    def simd_size_to_greg(self, *, reg : greg_base,
                          dt : adt) -> str:
        esfx = adt_size(dt)*8
        return self.asmwrap(f"vsetvli {reg}, zero, e{esfx}, m{self.lmul}, ta, ma")

    @property
    def c_simd_size_function(self):
        pre_oi = self.output_inline
        self.set_output_inline(yesno=True)
        result  = "inline size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap(f"vsetvli %[byte_size], zero, e8, m{self.lmul}, ta, ma")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        self.set_output_inline(yesno=pre_oi)
        return result

    def add_greg_voff(self, *, reg : greg_base, offset : int,
                      dt : adt) -> str:
        raise NotImplementedError(
                "RVV doesn't have an instruction to add a vector offset to a gp register")

    def zero_vreg(self, *, vreg : vreg_base, dt : adt) -> str:
        return self.asmwrap(f"vmv.v.i {vreg},0")


    @property
    def min_load_voff(self) -> int:
        return 0

    @property
    def max_load_voff(self) -> int:
        return 0

    def min_load_immoff(self, dt : adt) -> int:
        return 0

    def max_load_immoff(self, dt : adt) -> int:
        return 0


    def min_bcast_immoff(self, dt : adt) -> int:
        return self.min_load_immoff(dt)

    def max_bcast_immoff(self, dt : adt) -> int:
        return self.max_load_immoff(dt)

    @property
    def max_add_voff(self) -> int:
        return 0

    def vsetvli_keep_vl(self, *, dt : adt, lmul : int) -> str:
        """
        Change the element width and register grouping, keeping the vector
        length (vsetvli zero, zero): valid where SEW/LMUL stays the same, as
        from e32,m1 to e64,m2 for a widened register group
        """
        return self.asmwrap(f"vsetvli zero,zero,e{adt_size(dt) * 8},m{lmul},ta,ma")

    def greg_to_voffs(self, *, streg : greg_base, vreg : vreg_base, dt : adt) -> str:
        """
        An index vector of byte offsets 0, s, 2s... for indexed loads and
        stores (vluxei, vsuxei), the stride s in bytes in streg. For a
        constant stride vlse/vsse do the same without an index vector, but a
        gather is a choice of its own
        """
        return self.asmwrap(f"vid.v {vreg}") + self.asmwrap(f"vmul.vx {vreg},{vreg},{streg}")

    def max_mregs(self) -> int:
        # masked instructions take their mask in v0
        return 1

    def shared_registers(self) -> list[tuple[str, str, dict[int, int] | None]]:
        # the mask register is vector register 0
        return [("mreg", "vreg", {0: 0})]

    def fill_vector(self, *, sreg : freg_base,
                    vreg : vreg_base, dt : adt) -> str:
        return self.asmwrap(f"vfmv.v.f {vreg}, {sreg}")

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

    def vsetvli(self, *, vlreg : greg_base, avlreg : greg_base, dt : adt) -> str:
        """
        Set RVV vlen by requesting an AVL

        :param vlreg: GP register to write the resulting VLEN to
        :type vlreg: class:`asmgen.registers.greg_base`
        :param avlreg: GP register containing the requested AVL
        :type avlreg: class:`asmgen.registers.greg_base`
        :param dt: Data type of the elements
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: string with RVV instruction setting the VLEN
        :rtype: str
        """
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {vlreg}, {avlreg}, {dt_size}, m{self.lmul}, ta, ma")

    def vsetvlmax(self, *, reg : greg_base, dt : adt) -> str:
        """
        Set RVV vlen to maximum

        :param vlreg: GP register to write the resulting VLEN to
        :type vlreg: class:`asmgen.registers.greg_base`
        :param dt: Data type of the elements
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: string with RVV instruction setting the VLEN
        :rtype: str
        """
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {reg}, zero, {dt_size}, m{self.lmul}, ta, ma")

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
    def max_tregs(self, dt : adt) -> int:
        return 0

    rvv_no_tile_message = "RVV has no tiles (wait for IME/AME support)"

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError(self.rvv_no_tile_message)

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError(self.rvv_no_tile_message)

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError(self.rvv_no_tile_message)

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError(self.rvv_no_tile_message)

    def load_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        raise NotImplementedError("Lane loads not implemented for RVV")

    def store_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : adt) -> str:
        raise NotImplementedError("Lane stores not implemented for RVV")
