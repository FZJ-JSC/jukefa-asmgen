# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AArch64 SIMD opd3 base class for both NEON and SVE
"""

from typing import Callable,Any

from ....registers import (
    asm_data_type as adt,
    adt_size,
    adt_is_unsigned,
    data_reg
)
from ...op import (
    optransform,
    optransform_modifier as mod,
    operand_modifier as opd_mod,
    operation_signature
)
from ....registers import greg_base as _greg_base
from ...op.scalar import scalar_operands

class aarch64_simd_opd3_base(optransform):
    """
    NEON+SVE base opd3 implementation with methods shared by all
    opd3 operations
    """
    # the base ISA's operation for general registers (add, mul): a member
    # whose signatures are offered too and to which calls with them go
    scalar_class = None
    inst_base = "invalid"
    has_acc_suffix = False

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 dt_idxsuffixes : dict[adt,str]
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_idxsuffixes = dt_idxsuffixes

        self.signatures = []
        self.scalar_op = self.scalar_class(asmwrap=asmwrap) if self.scalar_class else None


    def all_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.all_signatures() if self.scalar_op else [])]

    def diagnose_failure(self,
                         modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        for name, mods in operand_modifiers.items():
            if opd_mod.ILANE in mods and f"{name}_lane" not in kwargs:
                raise ValueError(f"Argument missing: {name}_lane")
            if opd_mod.BLOCKLANE in mods and f"{name}_lane" not in kwargs:
                raise ValueError(f"Argument missing: {name}_lane")
            if opd_mod.BLOCKLANE in mods and f"{name}_blocksize" not in kwargs:
                raise ValueError(f"Argument missing: {name}_blocksize")
            if opd_mod.GLANE in mods:
                raise ValueError(f"{self.__class__.__name__} has no glane form")
            if opd_mod.VF in mods:
                raise ValueError(f"{self.__class__.__name__} has no VF form")

        if mod.PART in modifiers and 'part' not in kwargs:
            raise ValueError("Operand missing: part")

        a_dt = dts.get('adreg')
        c_dt = dts.get('cdreg')
        if a_dt and c_dt and adt_is_unsigned(a_dt):
            if adt_size(a_dt) == adt_size(c_dt):
                op_name = self.inst_base.upper()
                raise ValueError(
                    f"Only widening unsigned integer {op_name} supported in NEON"
                )


    def inst_prefix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        """
        Choose the correct instruction prefix based on specified types

        :param a_dt: Type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param c_dt: Type of the C component
        :type c_dt: class:`asmgen.registers.asm_data_type`
        :return: string containing the instruction prefix
        :rtype: str
        """
        if a_dt in [adt.FP8E5M2, adt.FP8E4M3, adt.FP16, adt.FP32, adt.FP64]:
            return "f"
        if a_dt in [adt.BF16]:
            return "bf"
        if a_dt in [adt.SINT8, adt.SINT16, adt.SINT32, adt.SINT64] and (a_dt == c_dt):
            return ""
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return "u"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return "s"

        raise RuntimeError("Unsupported datatype")

    def partial_inst_suffix(self, ways : int, part : int, **_):
        """
        Get the instruction suffix for a n-ways widening instruction computing
        the specified part

        :param ways: widening factor
        :type ways: int
        :param ways: which part to compute
        :type ways: int
        :return: string containing the required suffix
        :rtype: str
        """
        if ways.bit_count() != 1:
            raise ValueError(f"ways={ways} is not a power of 2")
        char_count = ways.bit_length()-1
        suf = 'l'*char_count

        # Works, but is too much of a 'detour' from the intent
        #suf += f"{part:b>{ways.bit_length()-1}b}".replace('1','t').replace('0','b')
        #return suf

        partsuf = ['b']*char_count
        for i in range(char_count):
            if (part >> i) & 0x1:
                partsuf[-(i+1)] = 't'

        suf += "".join(partsuf)

        return suf

    def build_operands(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> tuple[str,str,str]:
        """
        Build the strings for the 3 operands
        """

        # FMA/FMUL
        bdreg_mods = operand_modifiers.get('bdreg',set())
        if {opd_mod.ILANE, opd_mod.BLOCKLANE}.intersection(bdreg_mods):
            b = f"{bdreg}.{self.dt_idxsuffixes[b_dt]}[{kwargs['bdreg_lane']}]"
        else:
            b = f"{bdreg}.{self.dt_suffixes[b_dt]}"

        # FDOTA
        cdreg_mods = operand_modifiers.get('cdreg',set())
        if {opd_mod.ILANE, opd_mod.BLOCKLANE}.intersection(cdreg_mods):
            c = f"{cdreg}.{self.dt_idxsuffixes[c_dt]}[{kwargs['cdreg_lane']}]"
        else:
            c = f"{cdreg}.{self.dt_suffixes[c_dt]}"

        return (
            f"{adreg}.{self.dt_suffixes[a_dt]}",
            b,
            c)


    # ??? It's just 6?
    # pylint: disable-next=too-many-locals
    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg = None, cdreg : data_reg,
                       a_dt : adt, b_dt : adt = None, c_dt : adt,
                       modifiers : set[mod] = None,
                       operand_modifiers : dict[str,set[opd_mod]] = None,
                       **kwargs) -> str:

        # If scalar registers are passed, forward to the base ISA's operation
        if self.scalar_op is not None and isinstance(cdreg, _greg_base):
            return self.scalar_op(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                                  a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        if modifiers is None:
            modifiers = set()
        if operand_modifiers is None:
            operand_modifiers = dict()

        # This allows the SVE version to use the same codepath
        predicate = f"{kwargs['amreg']}/m," if 'amreg' in kwargs else ""

        # Add a/s suffix if instruction supports it
        suf = ""
        if self.has_acc_suffix:
            suf = "s" if mod.NP in modifiers else "a"

        inst = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt) +\
               self.inst_base + suf

        if mod.PART in modifiers:
            inst += self.partial_inst_suffix(
                    ways=adt_size(c_dt)//adt_size(a_dt),
                    part=kwargs['part'], a_dt=a_dt, c_dt=c_dt)

        a,b,c = self.build_operands(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                                    a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                                    operand_modifiers=operand_modifiers,
                                    **kwargs)


        return self.asmwrap(
            f"{inst} "
            f"{c},"
            f"{predicate}"
            f"{a},"
            f"{b}")
