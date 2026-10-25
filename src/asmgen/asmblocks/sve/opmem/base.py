# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE opmem operations
"""
import math
from typing import Callable,Any

from ...aarch64.opmem.base import aarch64_opmem
from ...aarch64.types import aarch64_greg, aarch64_freg
from ..types import sve_preg
from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod
)

from ....registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    adt_size,
    ait_size,
)

from .signatures import make_sve_opmem_signatures

class sve_opmem(opmem):
    """
    AArch64 SVE instruction with n data operands and 1 address operand.
    Inherits from opmem and composes aarch64_opmem for scalar routing.
    """

    bcast_supported : bool = False

    def __init__(self, action: opmem_action, asmwrap: Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap
        self.scalar_opmem = aarch64_opmem(action=action, asmwrap=asmwrap)

        self.signatures = make_sve_opmem_signatures(bcast_supported=self.bcast_supported)
        self.signatures.extend(self.scalar_opmem.all_signatures())

    def all_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Get instruction base string
        """
        return "ld" if self.action == opmem_action.LOAD else "st"


    def diagnose_mods(self, modifiers: set[mod]):
        """
        Diagnose if any modifiers are not supported at all
        """

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "SVE has no ld/st with 2D tile offset indices"),
            mod.TOFFSET: (ValueError, "SVE has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "SVE has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "SVE has no ld/st with GP-reg strides"),
            mod.NT:      (NotImplementedError, "Non-temporals for SVE not yet implemented"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

    def diagnose_opd_mods(self, opd_mods: dict[str,set[opd_mod]]):
        """
        Diagnose if any modifiers are not supported at all
        """

        unsupported_opd_mods = {
            opd_mod.ILANE : (ValueError, "SVE has no immediate lane ld/st"),
            opd_mod.ROW   : (ValueError, "SVE has no row selection ld/st"),
            opd_mod.COL   : (ValueError, "SVE has no column selection ld/st"),
            opd_mod.GLANE : (ValueError, "SVE has no GP-reg lane selection ld/st"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]) -> list[operation_signature]:

        self.diagnose_mods(modifiers)
        self.diagnose_opd_mods(opd_mods=operand_modifiers)

        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in operand_modifiers.items())
        if has_bcast and self.action != opmem_action.LOAD:
            raise ValueError("BCAST modifier is only valid for LOAD operations")

        if mod.VINDEX in modifiers and \
          (mod.VOFFSET in modifiers or mod.IOFFSET in modifiers):
            raise ValueError("VINDEX cannot be combined with IOFFSET/VOFFSET")

        required_params = {
            mod.STRUCT : ['nstructs'],
            mod.IOFFSET : ['ioffset'],
            mod.VOFFSET : ['voffset'],
            mod.GOFFSET : ['offreg'],
            mod.VINDEX : ['it','vidxreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")



    def get_mem_suffix(self, dt: adt) -> str:
        """
        Instruction data size suffix (e.g., ld1w for 32-bit words)
        """
        size = adt_size(dt)
        if size == 1:
            return "b"
        if size == 2:
            return "h"
        if size == 4:
            return "w"
        if size == 8:
            return "d"
        raise ValueError(f"Unsupported SVE memory size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        """
        Register data size suffix (e.g., z0.s for 32-bit singles)
        """
        size = adt_size(dt)
        if size == 1:
            return ".b"
        if size == 2:
            return ".h"
        if size == 4:
            return ".s"
        if size == 8:
            return ".d"
        raise ValueError(f"Unsupported SVE element size: {size}")

    def get_index_suffix(self, it: ait) -> str:
        """
        Register index size suffix (.d or .s)
        """
        size = ait_size(it)
        if size == 4:
            return ".s"
        if size == 8:
            return ".d"
        raise ValueError(f"Unsupported SVE index element size: {size}")

    def get_addressing(self, areg: aarch64_greg,
                       modifiers: set[mod], dt: adt, **kwargs) -> str:
        """
        Generate addressing string
        :param areg: GP reg containing base address
        :param modifiers: operation modifiers
        :param dt: data type to use
        """
        if not isinstance(areg, aarch64_greg):
            raise ValueError(f"{areg} is not an aarch64_greg")

        if mod.VINDEX in modifiers and mod.IOFFSET in modifiers:
            # gathers and scatters with vector bases: [z.d, #imm]
            lanes = ".s" if adt_size(dt) == 4 else ".d"
            return f"[{kwargs['vidxreg']}{lanes}, #{kwargs['ioffset']}]"

        if mod.VOFFSET in modifiers:
            return f"[{areg}, #{kwargs['voffset']}, MUL VL]"

        if mod.IOFFSET in modifiers:
            return f"[{areg}, #{kwargs['ioffset']}]"

        if mod.GOFFSET in modifiers:
            offreg = kwargs["offreg"]
            size = adt_size(dt)
            shift = ""
            if size > 1:
                lsl = int(math.log2(size))
                shift = f", lsl #{lsl}"
            return f"[{areg}, {offreg}{shift}]"

        if mod.VINDEX in modifiers:
            vidxreg = kwargs["vidxreg"]
            # Scatter/Gather: byte offsets - as RVV's (which can't scale),
            # AVX's (scale 1), greg_to_voffs's and the old methods' - in the
            # data's lanes, 32-bit ones sign-extended
            lanes = ".s" if adt_size(dt) == 4 else ".d"
            extend = ", sxtw" if ait_size(kwargs["it"]) == 4 else ""
            return f"[{areg}, {vidxreg}{lanes}{extend}]"

        return f"[{areg}]"

    # Wrong
    # pylint: disable-next=too-many-locals
    def implementation(self, *, dregs: list, agreg : aarch64_greg|None = None,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        if not dregs:
            raise ValueError("No dregs provided")

        a_dt = kwargs.pop('a_dt')

        # Forward scalars to AArch64 base
        if isinstance(dregs[0], (aarch64_greg, aarch64_freg)):
            return self.scalar_opmem(dregs=dregs, areg=agreg,
                                      dt=a_dt, modifiers=modifiers,
                                      operand_modifiers=operand_modifiers,
                                      **kwargs)

        # 1. Resolve Suffixes
        msuf = self.get_mem_suffix(a_dt)
        esuf = self.get_element_suffix(a_dt)

        # 2. Build Base Instruction (e.g. ld1w, ld2d, ld1rw)
        nstructs = kwargs.get("nstructs", 1)
        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in operand_modifiers.items())
        has_block = any(opd_mod.BLOCK in mods for _,mods in operand_modifiers.items())
        if has_bcast and has_block:
            inst = f"{self.inst_base}1rq{msuf}"
        elif has_bcast:
            inst = f"{self.inst_base}{nstructs}r{msuf}"
        else:
            inst = f"{self.inst_base}{nstructs}{msuf}"

        # 3. Resolve Predicate (Default to p0 if not passed)
        preg = kwargs.get("amreg", sve_preg(0))
        preg_str = f"{preg}/z" if self.action == opmem_action.LOAD else f"{preg}"

        # 4. Resolve Registers and Addressing
        dregs_str = ", ".join([f"{r}{esuf}" for r in dregs])
        addressing = self.get_addressing(agreg, modifiers, a_dt, **kwargs)

        return self.asmwrap(f"{inst} {{{dregs_str}}}, {preg_str}, {addressing}")
