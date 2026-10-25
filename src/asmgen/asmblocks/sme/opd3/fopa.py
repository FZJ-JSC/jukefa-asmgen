# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SME fopa implementation
"""

from typing import Callable, Any

from ....registers import (
    asm_data_type as adt,
    data_reg
)
from ...op import (
    opd3,
    opd3_modifier as mod,
    operation_signature,
    operand_modifier as opd_mod
)

from .signatures import make_sme_opd3_signatures

class sme_fopa(opd3):
    """
    SME ASM instructions of fused outer-product-accumulate operations
    """

    NIE_MESSAGE = "Not supported in SME"

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes

        self.signatures = make_sme_opd3_signatures(supports_np=True)

    def all_signatures(self) -> list[operation_signature]:
        return self.signatures

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        if mod.PART in modifiers:
            raise ValueError(("SME has no partial instructions "
                              "(widening instructions 'dot' neighbours)"))

        unsupported_opd_mods = {
            opd_mod.VF    : (ValueError, "SME has no VF opd3"),
            opd_mod.ILANE : (ValueError, "SME has no immediate lane opd3"),
            opd_mod.GLANE : (ValueError, "SME has no GP-reg lane selection opd3"),
            opd_mod.BCAST : (ValueError, "SME has no BCAST opd3"),
            opd_mod.ROW   : (ValueError, "SME has no row selection opd3"),
            opd_mod.COL   : (ValueError, "SME has no column selection opd3"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in operand_modifiers.items():
                if umod in mods:
                    raise exc_type(msg)

    def mopx_inst_str(self, a_dt : adt, b_dt : adt, suf : str) -> str:
        """
        Choose the correct base MOPX instruction based on specified types

        :param a_dt: Type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param suf: mop suffix (accumulate, subtract,...)
        :type suf: str
        :return: string containing the base instruction
        :rtype: str
        """
        if a_dt in [adt.FP8E5M2, adt.FP8E4M3, adt.FP16, adt.FP32, adt.FP64]:
            return f"fmop{suf}"
        if a_dt in [adt.BF16]:
            return f"bfmop{suf}"
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return f"umop{suf}"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return f"smop{suf}"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return f"sumop{suf}"
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return f"usmop{suf}"

        raise ValueError("Unsupported datatypes a={a_dt},b={b_dt}")

    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        suf = "s" if mod.NP in modifiers else "a"
        inst = self.mopx_inst_str(a_dt=a_dt, b_dt=b_dt, suf=suf)
        narrow_suf = self.dt_suffixes[a_dt]
        wide_suf = self.dt_suffixes[c_dt]
        return self.asmwrap(
            (f"{inst} {cdreg}.{wide_suf},"
             f"{kwargs['amreg']}/m,{kwargs['bmreg']}/m,"
             f"{adreg}.{narrow_suf},"
             f"{bdreg}.{narrow_suf}"))
