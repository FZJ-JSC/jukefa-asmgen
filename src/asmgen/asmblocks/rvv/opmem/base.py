# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 opmem base
"""

from typing import Any,Callable

from ....registers import (
    asm_data_type as adt,
    adt_size,
    ait_size,
    data_reg,
    greg_base
)

from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod
)

from ...riscv64.opmem.base import riscv64_opmem

from ..types import rvv_vreg
from ...riscv64.types import riscv64_greg,riscv64_freg


from .signatures import make_rvv_opmem_signatures

class rvv_opmem(opmem):
    """
    RVV instruction with 1 data operand and 1 address operand

    Abstraction for loads/stores (maybe also prefetches)
    """

    has_bcast = False

    def __init__(self, action : opmem_action,
                 asmwrap : Callable[[str],str],
                 lmul_getter :Callable[[],int]):
        self.action = action
        self.asmwrap = asmwrap
        self.get_lmul = lmul_getter

        self.scalar_opmem = riscv64_opmem(action=action, asmwrap=asmwrap)

        self.signatures = make_rvv_opmem_signatures(self.get_lmul, self.has_bcast)
        self.signatures.extend(self.scalar_opmem.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Instruction base
        """
        if self.action  == opmem_action.LOAD:
            return "vl"
        if self.action == opmem_action.STORE:
            return "vs"
        raise ValueError(f"Invalid action: {self.action}")

    def diagnose_unsupported_mods(self, modifiers: set[mod]):
        """
        Check if any modifier is unsupported at all
        """

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "RVV has no ld/st with 2D tile offset indices"),
            mod.IOFFSET: (ValueError, "RVV has no ld/st with immediate offsets"),
            mod.VOFFSET: (ValueError, "RVV has no ld/st with vector offsets"),
            mod.TOFFSET: (ValueError, "RVV has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "RVV has no ld/st with immediate strides"),
            mod.MASK:    (NotImplementedError, "RVV masked ld/st not implemented yet"),
            mod.POSTINC: (ValueError, "RVV has no postinc ld/st"),
            mod.NT:      (ValueError, "RVV has no non-temporals ld/st"),
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
            opd_mod.ROW:       (ValueError, "RVV has no row selection ld/st"),
            opd_mod.COL:       (ValueError, "RVV has no column selection ld/st"),
            opd_mod.ILANE:     (ValueError, "RVV has no immediate lane selection ld/st"),
            opd_mod.GLANE:     (ValueError, "RVV has no GP-reg lane selection ld/st"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_missing_params(self, modifiers : set[mod],
                                operand_modifiers: dict[str,set[opd_mod]],
                                kwargs : dict[str,Any]):
        """
        Check if mandatory additional parameters for specific modifiers are missing
        """

        # None that are supported need params
        del operand_modifiers

        required_params = {
            mod.STRUCT  : ['nstructs'],
            mod.GSTRIDE : ['streg'],
            mod.VINDEX  : ['it','vidxreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")

    def diagnose_bcast(self, modifiers : set[mod],
                       operand_modifiers: dict[str,set[opd_mod]]):
        """
        Check if BCAST modifier was used correctly
        """
        has_bcast = any(opd_mod.BCAST in mods for mods in operand_modifiers.values())

        if has_bcast:
            if self.action != opmem_action.LOAD:
                raise ValueError("BCAST modifier is only valid for LOAD operations")

            if not all(opd_mod.BCAST in mods
                       for name, mods in operand_modifiers.items()
                       if name != 'agreg'):
                raise ValueError(
                        "If an operand has BCAST modifier, all data operands must have it")

            if mod.GSTRIDE in modifiers:
                raise ValueError("BCAST and GSTRIDE are mutually exclusive")

        if (mod.GSTRIDE in modifiers) and (mod.VINDEX in modifiers):
            raise ValueError("mod.GSTRIDE and mod.VINDEX are mutually exclusive")


    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        self.diagnose_unsupported_mods(modifiers)
        self.diagnose_unsupported_opd_mods(operand_modifiers)

        if (mod.GSTRIDE in modifiers) and (mod.VINDEX in modifiers):
            raise ValueError("mod.GSTRIDE and mod.VINDEX are mutually exclusive")

        self.diagnose_bcast(modifiers, operand_modifiers)

        self.diagnose_missing_params(modifiers, operand_modifiers, kwargs)


    def get_instruction(self, base : str,
                        modifiers: set[mod],
                        operand_modifiers : dict[str,set[opd_mod]],
                        dt : adt, **kwargs) -> str:
        """
        Constructs the instruction mnemonic

        :param base: instruction base
        :param modifiers: operation modifiers
        :param dt: data type to use

        :return: string containing the mnemonic
        """

        inst_name = base

        # vl/vs
        # if mod.GSTRIDE: +s
        # if mod.VINDEX: +ux
        # if mod.STRUCT: +seg +{nf}
        # +e
        # if mod.VINDEX: +i
        # +{eew}

        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in operand_modifiers.items())

        if mod.GSTRIDE in modifiers or has_bcast:
            inst_name += "s"
        if mod.VINDEX in modifiers:
            inst_name += "ux" # only unordered.
        if mod.STRUCT in modifiers:
            nf = kwargs["nstructs"]
            inst_name += f"seg{nf}"
        inst_name += "e"
        if mod.VINDEX in modifiers:
            inst_name += "i"

        if mod.VINDEX in modifiers:
            # indexed: the EEW is the index's width (the data's is vtype's SEW)
            inst_name += str(ait_size(kwargs["it"])*8)
        else:
            inst_name += str(adt_size(dt)*8)

        inst_name += ".v"


        return inst_name

    def get_addressing(self, areg : riscv64_greg,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers

        :return: string containing the addressing
        """
        if not isinstance(areg, riscv64_greg):
            raise ValueError(f"{areg} is not a riscv64_greg")

        base_addr = f"({areg})"

        if mod.GSTRIDE in modifiers:
            streg = kwargs["streg"]
            if not isinstance(streg, riscv64_greg):
                raise ValueError(f"{streg} is not a riscv64_greg")
            return f"{base_addr}, {streg}"

        has_bcast = any(opd_mod.BCAST in mods
                        for _,mods in operand_modifiers.items())
        if has_bcast:
            return f"{base_addr}, zero"

        if mod.VINDEX in modifiers:
            vidxreg = kwargs["vidxreg"]
            if not isinstance(vidxreg, rvv_vreg):
                raise ValueError(f"{vidxreg} is not a rvv_vreg")
            return f"{base_addr}, {vidxreg}"

        return base_addr


    def implementation(self, *, dregs : list[data_reg],
                       agreg : greg_base|None = None,
                       modifiers : set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        if not dregs:
            raise ValueError("No dregs provided")

        a_dt = kwargs.pop('a_dt')

        # If scalar registers are passed, forward to base RISC-V
        if isinstance(dregs[0], (riscv64_greg, riscv64_freg)):
            return self.scalar_opmem(
                    dregs=dregs, areg=agreg, dt=a_dt,
                    modifiers=modifiers,
                    operand_modifiers=operand_modifiers,
                    **kwargs)

        inst = self.get_instruction(self.inst_base,
                                    modifiers,
                                    operand_modifiers,
                                    a_dt, **kwargs)
        addressing = self.get_addressing(agreg,
                                         modifiers,
                                         operand_modifiers,
                                         **kwargs)


        dreg_str = str(dregs[0])
        return self.asmwrap(f"{inst} {dreg_str}, {addressing}")
