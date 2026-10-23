# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX opd3 base case
"""
from abc import abstractmethod
from typing import Callable,Any

from ....registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    data_reg,
)
from ...op import (
    opd3,
    opd3_modifier as mod,
    operation_signature,
    operand_modifier as opd_mod
)

from ..types import reg_prefixer

from ....util import NIE_MESSAGE

from .signatures import make_avx_opd3_signatures
from ....registers import greg_base as _greg_base
from ...op.scalar import scalar_operands

class avx_opd3_base(opd3):
    """
    AVX base opd3 implementation with methods shared by all
    opd3 operations
    """
    # the base ISA's operation for general registers (add, mul): a member
    # whose signatures are offered too and to which calls with them go
    scalar_class = None

    supports_np = False

    # pylint: disable-next=too-many-positional-arguments
    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 it_suffixes : dict[ait,str],
                 rpref : reg_prefixer,
                 has_fp16 : bool = False,
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.it_suffixes = it_suffixes
        self.rpref = rpref
        self.has_fp16 = has_fp16

        self.signatures = make_avx_opd3_signatures(supports_np=self.supports_np,
                                                   has_fp16=self.has_fp16)
        self.scalar_op = self.scalar_class(asmwrap=asmwrap, rpref=rpref) if self.scalar_class else None

    def get_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.get_signatures() if self.scalar_op else [])]

    @abstractmethod
    def get_base_inst(self, modifiers : set[mod]) -> str:
        """
        Return the base instruction name based on the specified modifiers

        :param modifiers: set of modifiers to check the name for
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd3_modifier`]
        :return: ASM instruction name
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    def diagnose_unsupported_mods(self, modifiers: set[mod]):
        """
        Check if any modifier is unsupported at all
        """

        unsupported_mods = {
            mod.PART:  (ValueError, "AVX has no partial opd3 instructions"),
            mod.MASK:  (ValueError, "AVX has no masked opd3 instructions"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

    def diagnose_unsupported_opd_mods(self, opd_mods : dict[str,set[opd_mod]]):
        """
        Check if any operand modifier is unsupported at all
        """

        unsupported_opd_mods = {
            opd_mod.VF:        (ValueError, "AVX has no VF-form opd3"),
            opd_mod.BLOCKLANE: (ValueError, "AVX has no BLOCKLANE opd3"),
            opd_mod.BCAST:     (ValueError, "BCAST has no meaning for opd3"),
            opd_mod.ROW:       (ValueError, "AVX has no row selection opd3"),
            opd_mod.COL:       (ValueError, "AVX has no column selection opd3"),
            opd_mod.ILANE:     (ValueError, "AVX has no immediate lane selection opd3"),
            opd_mod.GLANE:     (ValueError, "AVX has no GP-reg lane selection opd3"),
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

    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg = None, cdreg : data_reg,
                       a_dt : adt, b_dt : adt = None, c_dt : adt,
                       modifiers : set[mod] = None,
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:

        # If scalar registers are passed, forward to the base ISA's operation
        if self.scalar_op is not None and isinstance(cdreg, _greg_base):
            return self.scalar_op(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                                  a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        if modifiers is None:
            modifiers = set()

        inst = self.get_base_inst(modifiers=modifiers)

        suf = 'p'+self.dt_suffixes[c_dt]
        pa = self.rpref(adreg)
        pb = self.rpref(bdreg)
        pc = self.rpref(cdreg)
        return self.asmwrap(f"{inst}{suf} {pa},{pb},{pc}")
