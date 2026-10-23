# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 opd3 base
"""
from abc import abstractmethod
from typing import Callable,Any

from ....registers import (
    asm_data_type as adt,
    adt_size,
    adt_is_float,adt_is_int,
    adt_is_signed,adt_is_unsigned,
    data_reg
)
from ...op import (
    opd3,
    opd3_modifier as mod,
    operation_signature,
    operand_modifier as opd_mod
)
from ....util import NIE_MESSAGE


from .signatures import make_rvv_opd3_signatures
from ....registers import greg_base as _greg_base
from ...op.scalar import scalar_operands

class rvv_opd3_base(opd3):
    """
    RVV 1.0 and 0.7.1 base opd3 implementation with methods shared by all
    opd3 operations
    """
    # the base ISA's operation for general registers (add, mul): a member
    # whose signatures are offered too and to which calls with them go
    scalar_class = None

    supports_np = False

    def __init__(self,
                 asmwrap : Callable[[str],str]):

        self.asmwrap = asmwrap
        self.operand_order = [2,0,1]

        self.signatures = make_rvv_opd3_signatures(supports_np=self.supports_np)
        self.scalar_op = self.scalar_class(asmwrap=asmwrap) if self.scalar_class else None

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
            mod.MASK:    (NotImplementedError, "RVV masked opd3 not implemented yet"),
            mod.PART:    (ValueError, "RVV does not use partial widening instructions"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

    def diagnose_unsupported_opd_mods(self, opd_mods : dict[str,set[opd_mod]]):
        """
        Check if any operand modifier is unsupported at all
        """

        unsupported_opd_mods = {
            opd_mod.BLOCKLANE: (ValueError, "RVV has no blocklane opd3"),
            opd_mod.BCAST:     (ValueError, "BCAST makes no sense with opd3"),
            opd_mod.ROW:       (ValueError, "RVV has no row-selection opd3"),
            opd_mod.COL:       (ValueError, "RVV has no column-seleciton opd3"),
            opd_mod.ILANE:     (ValueError, "RVV has no immediate lane selection opd3"),
            opd_mod.GLANE:     (ValueError, "RVV has no GP-reg lane selection opd3"),
        }
        for umod, (exc_type, msg) in unsupported_opd_mods.items():
            for _, mods in opd_mods.items():
                if umod in mods:
                    raise exc_type(msg)

    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        self.diagnose_unsupported_mods(modifiers)
        self.diagnose_unsupported_opd_mods(operand_modifiers)


    def inst_prefix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        """
        Return the first characters of the required instruction
        depending on the data type

        :param a_dt: Data type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Data type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param c_dt: Data type of the C component
        :type c_dt: class:`asmgen.registers.asm_data_type`
        :return: "vf" for FP types, "v" for INT types
        :rtype: str
        """
        _ = (b_dt, c_dt) # explicitly unused, possibly never relevant
        if adt_is_float(a_dt):
            return "vf"
        if adt_is_int(a_dt):
            return "v"

        raise RuntimeError("Unsupported datatype")

    def inst_suffix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        """
        Return the last characters of the required instruction
        depending on the data type

        :param a_dt: Data type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Data type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param c_dt: Data type of the C component
        :type c_dt: class:`asmgen.registers.asm_data_type`
        :return: "" for FP and signed INT types, "u" for unsigned INT types
            "su" for signed A and unsigned B, "us" for unsigned A and signed B
        :rtype: str
        """
        if adt_is_float(a_dt) and adt_is_float(b_dt) and adt_is_float(c_dt):
            return ""
        if adt_is_signed(a_dt) and adt_is_signed(b_dt) and adt_is_signed(c_dt):
            return ""
        if adt_is_unsigned(a_dt) and adt_is_unsigned(b_dt) and adt_is_unsigned(c_dt):
            return "u"
        if adt_is_signed(a_dt) and adt_is_unsigned(b_dt):
            return "su"
        if adt_is_unsigned(a_dt) and adt_is_signed(b_dt):
            return "us"

        raise RuntimeError("Unsupported datatype")


    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value,too-many-locals
    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg = None, cdreg : data_reg,
                       a_dt : adt, b_dt : adt = None, c_dt : adt,
                       modifiers : set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:



        # If scalar registers are passed, forward to the base ISA's operation
        if self.scalar_op is not None and isinstance(cdreg, _greg_base):
            return self.scalar_op(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                                  a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        pref = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        mix_pref = "w" if adt_size(c_dt)>adt_size(a_dt) else ""
        suf = self.inst_suffix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)

        b_mods = operand_modifiers.get('bdreg',set())
        if opd_mod.VF in b_mods and adt_is_int(b_dt):
            form_suf = "vx"
        elif opd_mod.VF in b_mods and adt_is_float(b_dt):
            form_suf = "vf"
        else:
            form_suf = "vv"

        base_inst = self.get_base_inst(modifiers=modifiers)

        inst = pref+mix_pref+base_inst+suf+"."+form_suf

        operands=[f"{adreg}",f"{bdreg}",f"{cdreg}"]

        operands_string = ','.join([operands[i] for i in self.operand_order])

        inst_str = f"{inst} {operands_string}"

        return self.asmwrap(inst_str)
