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
from ..riscv64.opd3 import riscv64_shift
from .opset import rvv_opset
from ..op.opset import opset_modifier as _opset_mod


# pylint: disable=too-many-public-methods

class rvv(riscv64):
    """
    RISC-V RVV 1.0 asmgen implementation
    """

    #: the mask population count (0.7.1: vmpopc.m)
    vpopc = "vcpop.m"
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
        self.shift = riscv64_shift(asmwrap=self.asmwrap)
        self.set = rvv_opset(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul)

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
        asmblock += self.asmwrap(f"{self.vpopc} {greg},{vreg2}")
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

    def add_greg_voff(self, *, reg : greg_base, offset : int,
                      dt : adt) -> str:
        raise NotImplementedError(
                "RVV doesn't have an instruction to add a vector offset to a gp register")

    def zero_vreg(self, *, vreg : vreg_base, dt : adt) -> str:
        return self.set(dreg=vreg, dt=dt, modifiers={_opset_mod.CONST}, constant="zero")


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

    def max_mregs(self) -> int:
        # masked instructions take their mask in v0
        return 1

    def shared_registers(self) -> list[tuple[str, str, dict[int, int] | None]]:
        # the mask register is vector register 0
        return [("mreg", "vreg", {0: 0})]

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

    def max_tregs(self, dt : adt) -> int:
        return 0

    rvv_no_tile_message = "RVV has no tiles (wait for IME/AME support)"

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError(self.rvv_no_tile_message)

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError(self.rvv_no_tile_message)
