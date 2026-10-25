# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base X86 loads/stores
"""

from typing import Callable, Any

from ...op import (
    opmem,
    opmem_modifier as mod,
    opmem_action,
    operation_signature,
    operand_modifier as opd_mod
)

from ....registers import asm_data_type as adt, adt_size
from ..types import x86_greg, avx_freg

from .signatures import make_x86_64_opmem_signatures
from ...op.opmem import param_signatures

class x86_opmem(opmem):
    """
    x86_64 scalar instruction with 1 data operand and 1 address operand.
    Handles standard scalar loads/stores (movb, movq, vmovss, vmovsd)
    """

    def __init__(self, action: opmem_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap
        self.rpref = rpref

        sigs = make_x86_64_opmem_signatures()
        self.signatures = sigs + param_signatures(sigs)

    def all_signatures(self) -> list[operation_signature]:
        return self.signatures

    def diagnose_unsupported_mods(self, modifiers: set[mod]):
        """
        Check if any modifier is unsupported at all
        """

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "Base X86_64 has no ld/st with 2D tile offset indices"),
            mod.VINDEX:  (ValueError, "Base X86_64 has no ld/st with 1D vector offset indices"),
            mod.VOFFSET: (ValueError, "Base X86_64 has no ld/st with 2D tile offsets"),
            mod.TOFFSET: (ValueError, "Base X86_64 has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "Base X86_64 has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "Base X86_64 has no ld/st with GP-reg strides"),
            mod.MASK:    (ValueError, "Base X86_64 has no masked ld/st"),
            mod.STRUCT:  (ValueError, "Base X86_64 has no structured ld/st"),
            mod.POSTINC: (ValueError, "Base X86_64 has no postinc ld/st"),
            mod.NT:      (ValueError, "Base X86_64 has no non-temporals ld/st"),
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
            opd_mod.BCAST:     (ValueError, "Base X86_64 has no broadcasting ld/st"),
            opd_mod.ROW:       (ValueError, "Base X86_64 has no row selection ld/st"),
            opd_mod.COL:       (ValueError, "Base X86_64 has no column selection ld/st"),
            opd_mod.ILANE:     (ValueError, "Base X86_64 has no immediate lane selection ld/st"),
            opd_mod.GLANE:     (ValueError, "Base X86_64 has no GP-reg lane selection ld/st"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]) -> list[operation_signature]:

        self.diagnose_unsupported_mods(modifiers)
        self.diagnose_unsupported_opd_mods(operand_modifiers)

        required_params = {
            mod.IOFFSET : ['ioffset'],
            mod.GOFFSET : ['offreg'],
            mod.SCALEOFFSET : ['offreg', 'scale_factor'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")

    def get_scalar_suffix(self, dt: adt, is_freg: bool) -> str:
        """
        Instruction suffix for int and fp scalars

        :param dt: data type
        :param dt: if an fp register is used
        """
        fp_sufs = {2:"sh",4:"ss",8:"sd"}
        i_sufs = {1:"b",2:"w",4:"l",8:"q"}
        size = adt_size(dt)
        if is_freg:
            return fp_sufs[size]
        return i_sufs[size]

    def get_addressing(self, areg: x86_greg, modifiers: set[mod], **kwargs) -> str:
        """
        Generate addressing string
        :param areg: GP reg containing base address
        :param modifiers: operation modifiers
        :param dt: data type to use
        """

        if mod.PARAM in modifiers:
            return f"%[{kwargs['param']}]"
        offset = kwargs.get("ioffset", 0) if mod.IOFFSET in modifiers else 0

        pareg = self.rpref(areg)
        register_part = f"{pareg}"
        if mod.GOFFSET in modifiers:
            poffreg = self.rpref(kwargs['offreg'])
            register_part += f",{poffreg}"
            if mod.SCALEOFFSET in modifiers:
                # the index register scaled: base + offreg x scale_factor
                register_part += f",{int(kwargs['scale_factor'])}"

        return f"{offset}({register_part})" if offset != 0 else f"({register_part})"

    def implementation(self, *, dregs: list, agreg : x86_greg|None = None,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        a_dt = kwargs['a_dt']
        dreg = dregs[0]

        is_freg = isinstance(dreg, avx_freg)
        base_inst = "vmov" if is_freg else "mov"
        inst = f"{base_inst}{self.get_scalar_suffix(a_dt, is_freg)}"

        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        pdreg = self.rpref(dreg)

        if self.action == opmem_action.LOAD:
            return self.asmwrap(f"{inst} {addressing}, {pdreg}")
        if self.action == opmem_action.STORE:
            return self.asmwrap(f"{inst} {pdreg}, {addressing}")

        raise ValueError(f"Action {self.action} not valid")
