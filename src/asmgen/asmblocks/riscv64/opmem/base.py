# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISCV64 +D/F opmem operations
"""

from typing import Callable,Any

from ....registers import (
    asm_data_type as adt,
    adt_size,
    data_reg,
    greg_base
)

from ...op import opmem,opmem_modifier as mod, opmem_action
from ...op import operation_signature
from ...op import operand_modifier as opd_mod
from ..types import riscv64_greg,riscv64_freg


from .signatures import make_riscv64_opmem_signatures
from ...op.opmem import param_signatures


class riscv64_opmem(opmem):
    """
    RISC-V 64 bit instructions with n data operand and 1 address operand

    Absraction for loads/stores (maybe also prefetches)
    """

    def __init__(self, action : opmem_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        sigs = make_riscv64_opmem_signatures()
        self.signatures = sigs + param_signatures(sigs)

    def diagnose_unsupported_mods(self, modifiers: set[mod]):
        """
        Check if any modifier is unsupported at all
        """

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "RISCV64 +D/F has no ld/st with 2D tile offset indices"),
            mod.VINDEX:  (ValueError, "RISCV64 +D/F has no ld/st with 1D vector offset indices"),
            mod.VOFFSET: (ValueError, "RISCV64 +D/F has no ld/st with 2D tile offsets"),
            mod.TOFFSET: (ValueError, "RISCV64 +D/F has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "RISCV64 +D/F has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "RISCV64 +D/F has no ld/st with GP-reg strides"),
            mod.MASK:    (ValueError, "RISCV64 +D/F has no masked ld/st"),
            mod.STRUCT:  (ValueError, "RISCV64 +D/F has no structured ld/st"),
            mod.POSTINC: (ValueError, "RISCV64 +D/F has no postinc ld/st"),
            mod.NT:      (ValueError, "RISCV64 +D/F has no non-temporals ld/st"),
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
            opd_mod.BCAST:     (ValueError, "RISCV64 +D/F has no broadcasting ld/st"),
            opd_mod.ROW:       (ValueError, "RISCV64 +D/F has no row selection ld/st"),
            opd_mod.COL:       (ValueError, "RISCV64 +D/F has no column selection ld/st"),
            opd_mod.ILANE:     (ValueError, "RISCV64 +D/F has no immediate lane selection ld/st"),
            opd_mod.GLANE:     (ValueError, "RISCV64 +D/F has no GP-reg lane selection ld/st"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_failure(self, modifiers: set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        self.diagnose_unsupported_mods(modifiers)
        self.diagnose_unsupported_opd_mods(operand_modifiers)

    @property
    def inst_base(self):
        """
        Instruction mnemonic base string
        """
        if self.action  == opmem_action.LOAD:
            return "l"
        if self.action == opmem_action.STORE:
            return "s"
        raise ValueError(f"Invalid action: {self.action}")

    def get_dt_suffix(self, dt : adt):
        """
        Instruction suffix based on data type

        :param dt: element data type
        """
        size_map = {1: "b", 2: "h", 4: "w", 8: "d", 16: "q"}
        return size_map[adt_size(dt)]

    def get_addressing(self, areg: riscv64_greg, modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers
        
        :return: string containing the addressing
        """
        if mod.PARAM in modifiers:
            return f"%[{kwargs['param']}]"
        if not isinstance(areg, riscv64_greg):
            raise ValueError(f"{areg} is not a riscv64_greg")

        offset = kwargs.get("ioffset", 0) if mod.IOFFSET in modifiers else 0
        return f"{offset}({areg})"

    def all_signatures(self) -> list[operation_signature]:
        return self.signatures

    def implementation(self, *, dregs : list[data_reg], agreg : greg_base|None = None,
                       modifiers : set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:


        a_dt = kwargs['a_dt']
        dreg = dregs[0]

        inst = self.inst_base
        if isinstance(dreg, riscv64_freg):
            inst = "f"+inst


        inst += self.get_dt_suffix(a_dt)
        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        return self.asmwrap(f"{inst} {dreg}, {addressing}")
