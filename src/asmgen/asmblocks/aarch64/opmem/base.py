# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Aarch64 opmem operations
"""

from typing import Callable

from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod,
)
from ...op.constraint import ValueType
from ....registers import asm_data_type as adt, adt_size

from ..types import aarch64_greg, aarch64_freg

from .signatures import make_aarch64_opmem_signatures
from ...op.opmem import param_signatures

class aarch64_opmem(opmem):
    """
    AArch64 scalar instruction with 1 data operand and 1 address operand.
    Handles standard scalar loads/stores (ldr, str, ldrb, strh, etc.)
    """

    def __init__(self, action: opmem_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap=asmwrap

        sigs = make_aarch64_opmem_signatures()
        self.signatures = sigs + param_signatures(sigs)

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Base root for the instruction
        """
        if self.action == opmem_action.LOAD:
            return "ldr"
        if self.action == opmem_action.STORE:
            return "str"
        raise ValueError(f"Invalid action: {self.action}")

    # I explicitly want it this way
    # pylint: disable-next=too-many-branches
    def diagnose_failure(self,
                         modifiers: set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,ValueType],
                         dts : dict[str,adt]):

        if mod.TINDEX in modifiers:
            raise ValueError(
                    "Base AArch64 has no ld/st with 2D tile offset indices")
        if mod.VINDEX in modifiers:
            raise ValueError(
                    "Base AArch64 has no ld/st with 1D vector offset indices")
        #if mod.GLANE in modifiers:
        #    raise ValueError("Base AArch64 has no GP-reg lane ld/st")
        #if mod.ILANE in modifiers:
        #    raise ValueError("Base AArch64 has no immediate lane ld/st")
        if mod.TOFFSET in modifiers:
            raise ValueError("Base AArch64 has no ld/st with 2D tile offsets")
        if mod.VOFFSET in modifiers:
            raise ValueError("Base AArch64 has no ld/st with vector offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError(
                    "Base AArch64 has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("Base AArch64 has no ld/st with GP-reg strides")
        if mod.STRUCT in modifiers:
            raise ValueError("Base AArch64 has no structured ld/st")
        #if mod.BCAST in modifiers:
        #    raise ValueError("Base AArch64 has no broadcasting ld/st")
        if mod.MASK in modifiers:
            raise ValueError("Base AArch64 has no masked ld/st")
        #if mod.ROW in modifiers:
        #    raise ValueError("Base AArch64 has no row selection ld/st")
        #if mod.COL in modifiers:
        #    raise ValueError("Base AArch64 has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for Base AArch64 not yet implemented")

        # `dts` has one entry per passed `dreg`; ld/st always use exactly one
        # register, data-less ops (prefetches) use none
        if self.action in (opmem_action.LOAD, opmem_action.STORE) \
                and len(dts) != 1:
            raise ValueError("AArch64 scalar load/store uses exactly one register.")

        if mod.POSTINC in modifiers:
            if "iinc" in kwargs and "increg" in kwargs:
                raise ValueError("iinc, increg are mutually exclusive")
            if "iinc" not in kwargs and "increg" not in kwargs:
                raise ValueError("Missing one of these parameters: iinc, increg")

        if mod.IOFFSET in modifiers and "ioffset" not in kwargs:
            raise ValueError("Missing operand: ioffset")

        if mod.GOFFSET in modifiers and "offreg" not in kwargs:
            raise ValueError("Missing operand: offreg")

    def get_addressing(self,
                       areg: aarch64_greg,
                       modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers
        
        :return: string containing the addressing
        """
        if mod.PARAM in modifiers:
            return f"%[{kwargs['param']}]"
        if not isinstance(areg, aarch64_greg):
            raise ValueError(f"{areg} is not an aarch64_greg")

        base_addr = f"[{areg}]"

        if mod.POSTINC in modifiers:
            increg = kwargs.get("increg", None)
            if increg is not None:
                return f"{base_addr}, {increg}"

            iinc = kwargs["iinc"]
            return f"{base_addr}, #{iinc}"

        if mod.IOFFSET in modifiers:
            ioffset = kwargs["ioffset"]
            if ioffset == 0:
                return base_addr
            return f"[{areg}, #{ioffset}]"

        if mod.GOFFSET in modifiers:
            offreg = kwargs["offreg"]
            return f"[{areg}, {offreg}]"

        return base_addr

    def get_inst_mnemonic(self, dt: adt, is_freg: bool) -> str:
        """
        Construct instruction mnemonic
        """
        base = self.inst_base

        if is_freg:
            return base

        size = adt_size(dt)
        if size == 1:
            return base + "b"
        if size == 2:
            return base + "h"

        return base

    def implementation(self, *, dregs: list, agreg : aarch64_greg|None = None,
                       modifiers: set[mod], **kwargs) -> str:

        if len(dregs) != 1:
            raise ValueError(
                    "AArch64 scalar load/store uses exactly one register.")

        a_dt = kwargs['a_dt']
        dreg = dregs[0].retype(dt=a_dt)
        is_freg = isinstance(dreg, aarch64_freg)

        inst = self.get_inst_mnemonic(a_dt, is_freg)
        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        return self.asmwrap(f"{inst} {dreg}, {addressing}")
