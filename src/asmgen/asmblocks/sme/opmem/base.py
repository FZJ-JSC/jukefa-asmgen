# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
opmem base class for SME
"""
import math
from typing import Callable, Any

from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod
)

from ....registers import asm_data_type as adt, adt_size

from ...aarch64.types import aarch64_greg
from ..types import sme_treg
from ...sve.opmem import sve_load,sve_store

from .signatures import make_sme_opmem_signatures


class sme_opmem(opmem):
    """
    AArch64 SME instruction generator.
    Handles ZA tile slices (sme_treg) and SME2 Non-Temporal loads/stores.
    Routes all other operations to sve_opmem.
    """

    _SVE_DELEGATE = {
        opmem_action.STORE : sve_store,
        opmem_action.LOAD : sve_load,
    }

    def __init__(self, action: opmem_action, asmwrap: Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        # fix for bcast_supported in the SVE delegate
        self.sve_opmem = self._SVE_DELEGATE[action](asmwrap=asmwrap)

        self.signatures = make_sme_opmem_signatures()
        self.signatures.extend(self.sve_opmem.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Instruction string base root
        """
        return "ld" if self.action == opmem_action.LOAD else "st"

    # It's fine, but add subdiagnosis methods if anything more gets added
    # pylint: disable-next=too-many-locals,too-many-branches
    def diagnose_failure(self, modifiers: set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        unsupported_mods = {
            mod.GSTRIDE: (ValueError, "SME has no ld/st with GP-reg strides"),
            mod.IOFFSET: (ValueError, "SME has no ld/st with immediate element offsets"),
            mod.ISTRIDE: (ValueError, "SME has no ld/st with immediate strides"),
            mod.POSTINC: (ValueError, "SME has no postinc ld/st"),
            mod.VINDEX:  (ValueError, "SME has no gather/scatter"),
            mod.TINDEX:  (ValueError, "SME has no ld/st with 2D tile offset indices"),
            mod.TOFFSET: (ValueError, "SME has no ld/st with 2D tile offsets"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)


        unsupported_opd_mods = {
            opd_mod.ILANE : (ValueError, "SME has no immediate lane ld/st"),
            opd_mod.GLANE : (ValueError, "SME has no GP-reg lane ld/st"),
            opd_mod.BCAST : (ValueError, "SME has no broadcasting ld"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for opd, mods in operand_modifiers.items():
                if umod in mods:
                    raise exc_type(msg)

        has_row = any(opd_mod.ROW in mods
                      for _,mods in operand_modifiers.items())
        has_col = any(opd_mod.COL in mods
                      for _,mods in operand_modifiers.items())

        if has_row and has_col:
            raise ValueError("ROW and COL modifiers are mutually exclusive")

        if mod.NT in modifiers and (has_row or has_col):
            raise ValueError(("Non-temporal strided operations cannot be combined "
                              "with tile slice modifiers (ROW/COL)"))

        if has_row or has_col:
            if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
                raise ValueError(("SME tile slice operations only support scalar+scalar "
                                 "(GOFFSET) memory addressing"))

        addr_mods = {mod.IOFFSET, mod.VOFFSET, mod.GOFFSET}.intersection(modifiers)
        if len(addr_mods) > 1:
            raise ValueError(f"Mutually exclusive addressing modifiers used: {addr_mods}")


        required_params = {
            mod.STRUCT : ['nstructs'],
            mod.IOFFSET : ['ioffset'],
            mod.VOFFSET : ['voffset'],
            mod.GOFFSET : ['offreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")

        opd_mod_required_params = {
            opd_mod.ROW : ['rowreg','immrow'],
            opd_mod.COL : ['colreg','immcol'],
        }
        for opd, mods in operand_modifiers.items():
            for m, plist in opd_mod_required_params.items():
                for p in plist:
                    opd_p = f"{opd}_{p}"
                    if m in mods and opd_p not in kwargs:
                        raise ValueError(
                                f"{m.name} modifier for {opd} requires '{opd_p}' parameter")


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
        raise ValueError(f"Unsupported SME memory size: {size}")

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
        raise ValueError(f"Unsupported SME element size: {size}")

    def get_addressing(self, areg: aarch64_greg, modifiers: set[mod], dt: adt, **kwargs) -> str:
        """
        Generate addressing string
        :param areg: GP reg containing base address
        :param modifiers: operation modifiers
        :param dt: data type to use
        """
        # Standardizes addressing for SME-specific instructions
        if mod.VOFFSET in modifiers:
            return f"[{areg}, #{kwargs['voffset']}, MUL VL]"

        if mod.IOFFSET in modifiers:
            return f"[{areg}, #{kwargs['ioffset']}]"

        if mod.GOFFSET in modifiers:
            offreg = kwargs["offreg"]
            size = adt_size(dt)
            if size > 1:
                lsl = int(math.log2(size))
                return f"[{areg}, {offreg}, lsl #{lsl}]"
            return f"[{areg}, {offreg}]"

        return f"[{areg}]"

    # Inlining any params or breaking the method up IMHO doesn't impove readability
    # pylint: disable-next=too-many-locals
    def implementation(self, *, dregs: list, agreg : aarch64_greg|None = None,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        a_dt = kwargs.pop('a_dt')
        # dts is rebuilt by the forwarded opmem.__call__
        kwargs.pop('dts', None)

        # --- ROUTING LOGIC ---
        # If it's not a Tile Register AND it's not a Non-Temporal instruction, SVE handles it.
        if not isinstance(dregs[0], sme_treg) and mod.NT not in modifiers:
            return self.sve_opmem(dregs=dregs, areg=agreg, dt=a_dt,
                                   modifiers=modifiers,
                                   operand_modifiers=operand_modifiers,
                                   **kwargs)

        msuf = self.get_mem_suffix(a_dt)
        esuf = self.get_element_suffix(a_dt)
        addressing = self.get_addressing(agreg, modifiers, a_dt, **kwargs)

        # 1. Tile Slice (ZA)
        if isinstance(dregs[0], sme_treg):

            has_row = any(opd_mod.ROW in mods
                          for _,mods in operand_modifiers.items())
            # Implicitly,through signature, if it's not has_row, it has col
            #has_col = any(opd_mod.COL in mods
            #              for _,mods in operand_modifiers.items())

            hv = "h" if has_row else "v"
            idx_reg = kwargs["adreg_rowreg"] if has_row else kwargs["adreg_colreg"]
            slice_imm = kwargs["adreg_immrow"] if has_row else kwargs["adreg_immcol"]

            # e.g., za0h.d[w12, 0]
            dregs_str = f"{dregs[0]}{hv}{esuf}[{idx_reg.get_wreg()}, {slice_imm}]"
            inst = f"{self.inst_base}1{msuf}"

            preg = kwargs["amreg"]
            preg_str = f"{preg}/z" if self.action == opmem_action.LOAD else f"{preg}"

        elif mod.NT in modifiers:
            dregs_str = ", ".join([f"{r}{esuf}" for r in dregs])
            inst = f"{self.inst_base}nt1{msuf}"

            # Predicate-as-counter (PN8-PN15) validation and string generation
            preg = kwargs["amreg"]
            preg_str = f"{preg}/z" if self.action == opmem_action.LOAD else f"{preg}"
        else:
            raise ValueError("Unhandled SME instruction configuration")

        return self.asmwrap(f"{inst} {{{dregs_str}}}, {preg_str}, {addressing}")
