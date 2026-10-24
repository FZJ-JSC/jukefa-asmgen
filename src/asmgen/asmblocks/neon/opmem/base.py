# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD operations with n data operand and 1 address operand (load/store)
"""

from typing import Callable,Any

from ...aarch64.opmem.base import aarch64_opmem
from ...aarch64.types import aarch64_greg,aarch64_freg
from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod,
    make_ord_prefix
)
from ....registers import asm_data_type as adt, adt_size

from .signatures import make_neon_opmem_signatures

class neon_opmem(opmem):
    """
    NEON (ASIMD) instruction with n data operands and 1 address operand.
    Uses aarch64_opmem to automatically handle scalar routing.
    """

    bcast_supported = False

    def __init__(self, action: opmem_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        self.scalar_opmem = aarch64_opmem(action=action, asmwrap=asmwrap)

        self.signatures = make_neon_opmem_signatures(
                bcast_supported=self.bcast_supported)
        self.signatures.extend(self.scalar_opmem.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Get base root of instruction
        """
        if self.action == opmem_action.LOAD:
            return "ld"
        if self.action == opmem_action.STORE:
            return "st"
        raise ValueError(f"Invalid action: {self.action}")


    def diagnose_unsupported_mods(self, modifiers: set[mod]):
        """
        Check if any modifier is unsupported at all
        """

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "NEON has no ld/st with 2D tile offset indices"),
            mod.VINDEX:  (ValueError, "NEON has no ld/st with 1D vector offset indices"),
            #mod.GLANE:   (ValueError, "NEON has no GP-reg lane ld/st"),
            mod.TOFFSET: (ValueError, "NEON has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "NEON has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "NEON has no ld/st with GP-reg strides"),
            mod.MASK:    (ValueError, "NEON has no masked ld/st"),
            #mod.ROW:     (ValueError, "NEON has no row selection ld/st"),
            #mod.COL:     (ValueError, "NEON has no column selection ld/st"),
            mod.NT:      (NotImplementedError, "Non-temporals for NEON not yet implemented"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

    def diagnose_unsupported_opd_mods(self, opd_mods : dict[str,set[opd_mod]]):
        """
        Check if any operand modifier is unsupported at all
        """

        unsupported_opd_mods = {
            opd_mod.VF:        (ValueError, "VF mod makes no sense for ld/st"),
            opd_mod.BLOCKLANE: (ValueError, "BLOCKLANE makes no sense for ld/st"),
            opd_mod.ROW:       (ValueError, "NEON has no row selection ld/st"),
            opd_mod.COL:       (ValueError, "NEON has no column selection ld/st"),
            opd_mod.GLANE:     (ValueError, "NEON has no GP-reg lane selection ld/st")
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_bcast(self, opd_mods : dict[str,set[opd_mod]]):
        """
        Check if BCAST modifier was used correctly
        """

        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in opd_mods.items())
        has_ilane = any(opd_mod.ILANE in mods
                        for _,mods in opd_mods.items())
        if has_bcast:
            if self.action != opmem_action.LOAD:
                raise ValueError("BCAST modifier is only valid for LOAD operations")
            if has_ilane:
                raise ValueError("BCAST cannot be combined with ILANE")

            # All operands must have BCAST
            if not all(opd_mod.BCAST in mods for _,mods in opd_mods.items()):
                raise ValueError(
                        "IF an operand has BCAST modifier, all operands must have it")

    def diagnose_ilane(self,
                       modifiers : set[mod],
                       opd_mods : dict[str,set[opd_mod]],
                       kwargs : dict[str,Any]):
        """
        Check if ILANE modifier was used correctly
        """

        has_ilane = any(opd_mod.ILANE in mods
                        for _,mods in opd_mods.items())
        if has_ilane:
            # All operands must have ILANE
            if not all(opd_mod.ILANE in mods for _,mods in opd_mods.items()):
                raise ValueError(
                        "IF an operand has ILANE modifier, all operands must have it")

            # All operands must have lane specified
            nstructs = 1
            if mod.STRUCT in modifiers:
                nstructs = kwargs['nstructs']
            for i in range(nstructs):
                dregname = f"{make_ord_prefix(i)}dreg"
                if f"{dregname}_lane" not in kwargs:
                    raise ValueError(
                            f"ILANE modifier requires '{dregname}_lane' parameter")

            # All lanes must be the same
            if nstructs > 1:
                adreg_lane = kwargs['adreg_lane']

                for i in range(nstructs):
                    dregname = f"{make_ord_prefix(i)}dreg"
                    if kwargs[f"{dregname}_lane"] != adreg_lane:
                        raise ValueError(
                                ("NEON STRUCT+ILANE ld/st requires the same lane "
                                 " to be used for all dregs"
                                ))

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        self.diagnose_unsupported_mods(modifiers)
        self.diagnose_unsupported_opd_mods(opd_mods=operand_modifiers)


        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in operand_modifiers.items())
        has_ilane = any(opd_mod.ILANE in mods
                        for _,mods in operand_modifiers.items())



        if {mod.VOFFSET, mod.IOFFSET} & modifiers:
            if mod.STRUCT in modifiers or has_bcast or has_ilane:
                raise ValueError(
                        ("VOFFSET/IOFFSET cannot be combined with STRUCT, "
                         "ILANE, or BCAST"))

        self.diagnose_bcast(opd_mods=operand_modifiers)


        if mod.POSTINC in modifiers:
            has_iinc = "iinc" in kwargs
            has_increg = "increg" in kwargs

            if has_iinc and has_increg:
                raise ValueError("iinc, increg are mutually exclusive")
            if not has_iinc and not has_increg:
                raise ValueError("Missing one of these parameters: iinc, increg")

        if mod.STRUCT in modifiers and "nstructs" not in kwargs:
            raise ValueError("STRUCT modifier requires 'nstructs' parameter")

        self.diagnose_ilane(modifiers=modifiers,
                            opd_mods=operand_modifiers,
                            kwargs=kwargs)


    def get_arrangement(self, dt: adt) -> str:
        """
        NEON suffixes for full simd registers
        """
        size = adt_size(dt)
        if size == 1:
            return ".16b"
        if size == 2:
            return ".8h"
        if size == 4:
            return ".4s"
        if size == 8:
            return ".2d"
        raise ValueError(f"Unsupported NEON vector size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        """
        NEON suffixes for lanes
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
        raise ValueError(f"Unsupported NEON lane size: {size}")

    def get_addressing(self, areg: aarch64_greg,
                       modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers
        
        :return: string containing the addressing
        """

        base_addr = f"[{areg}]"

        if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
            byte_offset = kwargs.get("ioffset", 0)
            if mod.VOFFSET in modifiers:
                byte_offset = kwargs["voffset"] * 16
            if byte_offset == 0:
                return base_addr
            return f"[{areg}, #{byte_offset}]"

        if mod.POSTINC in modifiers:
            increg = kwargs.get("increg")
            if increg is not None:
                if not isinstance(increg, aarch64_greg):
                    raise ValueError(f"{increg} is not an aarch64_greg")
                return f"{base_addr}, {increg}"
            iinc = kwargs["iinc"]
            return f"{base_addr}, #{iinc}"

        return base_addr

    # I don't know how pylint counts it, but it's not 16
    # pylint: disable-next=too-many-locals
    def implementation(self, *,
                       dregs: list, agreg : aarch64_greg|None = None,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        if not dregs:
            raise ValueError("No dregs provided")

        a_dt = kwargs.pop('a_dt')

        # If scalar registers are passed, forward to base AArch64
        if isinstance(dregs[0], (aarch64_greg, aarch64_freg)):
            return self.scalar_opmem(dregs=dregs, areg=agreg, dt=a_dt,
                                      modifiers=modifiers,
                                      operand_modifiers=operand_modifiers,
                                      **kwargs)

        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
            inst = "ldr" if self.action == opmem_action.LOAD else "str"
            return self.asmwrap(f"{inst} q{dregs[0].idx}, {addressing}")

        nstructs = kwargs.get("nstructs",1)

        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in operand_modifiers.items())

        inst = f"{self.inst_base}{nstructs}r" if has_bcast \
                else f"{self.inst_base}{nstructs}"

        has_ilane = any(opd_mod.ILANE in mods
                        for _,mods in operand_modifiers.items())

        if has_ilane:
            # Grab the first one as all have to be the same value
            lane = kwargs["adreg_lane"]
            arrangement = self.get_element_suffix(a_dt)
            reg_list_str = ", ".join([f"{r}{arrangement}" for r in dregs])
            dreg_str = f"{{{reg_list_str}}}[{lane}]"
        else:
            arrangement = self.get_arrangement(a_dt)
            reg_list_str = ", ".join([f"{r}{arrangement}" for r in dregs])
            dreg_str = f"{{{reg_list_str}}}"

        return self.asmwrap(f"{inst} {dreg_str}, {addressing}")
