# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SME asm generator
"""

from ...registers import (
    asm_data_type as adt,
    reg_tracker,
    adt_size,
    treg_base, vreg_base, greg_base,
)
from ..sve import sve

from .types import sme_treg
from .opd3 import sme_fopa
from .move import sme_move

from .opmem.load import sme_load
from .opmem.store import sme_store


class sme(sve):
    """
    SME asmgen implementation
    """
    # SVE's, and SME's own: FEAT_SME2, FEAT_SME_F64F64, _I16I64, _F16F16,
    # _B16B16, _F8F16, _F8F32
    FEATURES = sve.FEATURES + ("SME2", "F64F64", "I16I64", "F16F16", "SME_B16B16",
                               "F8F16", "F8F32")


    def isa_setup(self, predicate : int = 0):
        """
        What this ISA needs set up before a kernel and torn down after

        :param predicate: predicate register for the all-true predicate
        """
        from ..setup import sme_setup
        return sme_setup(self, predicate=predicate)

    def __init__(self):
        super().__init__()
        self.valid_rcregs = [str(self.greg(i)) for i in range(12,16)]
        self.fopa = sme_fopa(asmwrap=self.asmwrap,
                             dt_suffixes=self.dt_suffixes)

        self.load = sme_load(asmwrap = self.asmwrap)
        self.store = sme_store(asmwrap = self.asmwrap)

        self.move = sme_move(asmwrap=self.asmwrap,
                                  dt_suffixes=self.dt_suffixes)

    @property
    def c_simd_size_function(self):
        pre_oi = self.output_inline
        self.set_output_inline(yesno=True)
        result  = "inline size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap("smstart")
        result += "        "+self.asmwrap("mov %[byte_size],#0")
        result += "        "+self.asmwrap("incb %[byte_size]")
        result += "        "+self.asmwrap("smstop")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        self.set_output_inline(yesno=pre_oi)
        return result

    def max_tregs(self, dt : adt) -> int:
        return adt_size(dt)
    def max_mregs(self):
        return 16

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        return sme_treg(reg_idx, dt)

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        # the tiles to zero are a list: zero {za0.d}
        return self.asmwrap(f"zero {{{treg}.{suf}}}")


    def insert_tile_rows(self, *,
                       rreg : greg_base,
                       roff_start : int, roff_end : int,
                       treg : treg_base,
                       vregs : list[vreg_base],
                       dt :adt) -> str:
        """
        Returns ASM string for inserting one or multiple vregs as rows into the treg

        :param rreg: register containing the starting row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff_start: row offset to base row to start inserting at
        :type roff_start: int
        :param roff_end: row offset to base row to stop inserting at
        :type roff_end: int
        :param treg: tile register to insert rows into
        :type treg: class:`asmgen.register.treg_base`
        :param vregs: list of vector registers containing the rows to insert
        :type vregs: list[class:`asmgen.register.treg_base`]
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        suf = self.dt_suffixes[dt]
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"
                              f"(allowed: {','.join(self.valid_rcregs)})"))

        vsrc = "{" + ", ".join([f"{vreg}.{suf}" for vreg in vregs]) + "}"

        opt_preg = ""
        roff_str = f"{roff_start}:{roff_end}"
        if 1 == len(vregs):
            opt_preg = ",p0/m"
            roff_str = f"{roff_start}"

        return self.asmwrap(
            f"mov {treg}h.{suf}[{rreg},{roff_str}]{opt_preg},{vsrc}")

    def extract_tile_rows(self, *,
                       rreg : greg_base,
                       roff_start : int, roff_end : int,
                       treg : treg_base,
                       vregs : list[vreg_base],
                       dt :adt):
        """
        Returns ASM string for extracting one or multiple rows from the treg into
        the specified vregs

        :param rreg: register containing the starting row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff_start: row offset to base row to start extracting at
        :type roff_start: int
        :param roff_end: row offset to base row to stop extracting at
        :type roff_end: int
        :param treg: tile register to extract rows from
        :type treg: class:`asmgen.register.treg_base`
        :param vregs: list of vector registers to write the rows to
        :type vregs: list[class:`asmgen.register.treg_base`]
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        suf = self.dt_suffixes[dt]
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"
                              f" (allowed: {','.join(self.valid_rcregs)})"))

        vdest = "{" + ", ".join([f"{vreg}.{suf}" for vreg in vregs]) + "}"

        opt_preg = ""
        roff_str = f"{roff_start}:{roff_end}"
        if 1 == len(vregs):
            opt_preg = ",p0/m"
            roff_str = f"{roff_start}"

        return self.asmwrap(
            f"mov {vdest}{opt_preg},{treg}h.{suf}[{rreg},{roff_str}]")


    def load_tile_row(self, *,
                       areg : greg_base,
                       rreg : greg_base,
                       roff : int,
                       voff : int,
                       treg : treg_base,
                       dt :adt) -> str:
        """
        Returns ASM string for loading a row of data into a treg

        :param areg: GP register containing the base memory address
        :type areg: class:`asmgen.register.greg_base`

        :param rreg: register containing the base row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff: row offset to base row to load the data into
        :type roff: int
        :param voff: offset to base address in number of vectors
        :type voff: int
        :param treg: tile register to load the row into
        :type treg: class:`asmgen.register.treg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"

                              f" (allowed: {','.join(self.valid_rcregs)})"))

        rreg_str = str(rreg).replace('x','w')
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        address = f"[{areg}]"
        if voff > 0:
            address = f"[{areg}, #{voff}, MUL VL]"
        return self.asmwrap(
            f"ld1{msuf} {{{treg}h.{suf}[{rreg_str},{roff}]}},p0/z,{address}")

    def store_tile_row(self, *,
                       areg : greg_base,
                       rreg : greg_base,
                       roff : int,
                       voff : int,
                       treg : treg_base,
                       dt :adt):
        """
        Returns ASM string for storing a row of data from a treg into memory

        :param areg: GP register containing the base memory address
        :type areg: class:`asmgen.register.greg_base`

        :param rreg: register containing the base row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff: row offset to base row to store the data from
        :type roff: int
        :param voff: offset to base address in number of vectors
        :type voff: int
        :param treg: tile register to store the row from
        :type treg: class:`asmgen.register.treg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"
                              f" (allowed: {','.join(self.valid_rcregs)})"))

        rreg_str = str(rreg).replace('x','w')
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        address = f"[{areg}]"
        if voff > 0:
            address = f"[{areg}, #{voff}, MUL VL]"
        return self.asmwrap(
            f"st1{msuf} {{{treg}h.{suf}[{rreg_str},{roff}]}},p0/m,{address}")

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base, dt : adt) -> str:
        raise NotImplementedError(
            "SME has no tile loading instruction, use load_vector* and insert_tile_row/column")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base, dt : adt) -> str:
        raise NotImplementedError(
            "SME has no tile storing instruction, use extract_tile_row/column and store_vector*")
