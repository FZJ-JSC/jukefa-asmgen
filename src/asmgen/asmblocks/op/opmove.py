# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation moving data between data registers
"""

from enum import auto

from abc import abstractmethod

from .operation import operation
from .signature import operation_signature
from .modifier import operation_modifier
from .operand import operand_modifier
from .misc import make_ord_prefix

from ...registers import (
    asm_data_type as adt,
    data_reg,
)

# Modifiers are mostly on operands (ROW,LANE, etc...)
class opmove_modifier(operation_modifier):
    """
    Possible modifiers for an instruction/operation
    """
    MASK = auto()
    MULTIPLE_IN = auto()
    MULTIPLE_OUT = auto()
    BITS = auto()      # the bits as they are, between register files (greg to freg): no conversion
    PERMUTE = auto()   # two sources' units interleaved, unzipped or transposed (pattern, bits, segment)


#: what a permute does with its sources' units (bits wide), within each segment:
#:   zip_lo   A0 B0 A1 B1 ... (the low halves interleaved)   zip_hi   the high halves
#:   uzp_even A0 A2 ... B0 B2 ... (the even units of each)  uzp_odd  the odd units
#:   trn_even A0 B0 A2 B2 ... (pairs transposed)             trn_odd  A1 B1 A3 B3 ...
PERMUTE_PATTERNS = ("zip_lo", "zip_hi", "uzp_even", "uzp_odd", "trn_even", "trn_odd")


def permuted(pattern : str, a : list, b : list, *, element_bits : int, bits : int,
             segment : int | None = None) -> list:
    """
    What a permute makes of a and b, two vectors' elements in order: the
    pattern on units of bits, within each segment of segment bits - None, the
    whole vector. A permute's meaning, for networks built of them and tests.
    """
    per_unit = bits // element_bits
    seg = len(a) if segment is None else segment // element_bits
    out = []
    for s in range(0, len(a), seg):
        ua = [a[s + i:s + i + per_unit] for i in range(0, seg, per_unit)]
        ub = [b[s + i:s + i + per_unit] for i in range(0, seg, per_unit)]
        n = len(ua)
        units = {"zip_lo": [x for i in range(n // 2) for x in (ua[i], ub[i])],
                 "zip_hi": [x for i in range(n // 2, n) for x in (ua[i], ub[i])],
                 "uzp_even": ua[0::2] + ub[0::2],
                 "uzp_odd": ua[1::2] + ub[1::2],
                 "trn_even": [x for i in range(0, n, 2) for x in (ua[i], ub[i])],
                 "trn_odd": [x for i in range(1, n, 2) for x in (ua[i], ub[i])]}[pattern]
        out += [e for unit in units for e in unit]
    return out


def permute_signatures(forms) -> list:
    """
    PERMUTE: adreg and bdreg the sources, cdreg the destination - one
    signature per (data type, pattern, bits, segment) of forms
    """
    from .operand import operand_shape as osh, operand_type as ot, operand_role as orl, register_type as rgt
    return [operation_signature(modifiers={opmove_modifier.PERMUTE},
                                structural_params={"pattern": pattern, "bits": bits, "segment": segment},
                                operands={name: osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt)
                                          for name in ("adreg", "bdreg", "cdreg")})
            for dt, pattern, bits, segment in forms]


class opmove(operation):
    """
    Assembly/IR instruction moving data between data registers

    Absraction for moves,lane extracts/inserts, row/col extracts/inserts,
    bcasts, converts/casts, sign extends, etc...
    """
    NIE_MESSAGE="Method not implemented"

    def __call__(self, *,
                 dregs : list[data_reg],
                 dts : list[adt],
                 modifiers : set[opmove_modifier] = None,
                 operand_modifiers : dict[str,set[operand_modifier]] = None,
                 **kwargs) -> str:
        """
        Return the ASM/IR instruction
        
        :param dregs : Data registers
        :type dregs : list[class:`asmgen.registers.data_reg`]
        :param areg : Address register
        :type areg : class:`asmgen.registers.greg_base`
        :param dt : Data type
        :type dt : class:`asmgen.registers.asm_data_type`
        :return : ASM/IR instruction corresponding to the operation
        :rtype : str
        """

        if modifiers is None:
            modifiers = set()
        if operand_modifiers is None:
            operand_modifiers = dict()

        return self.execute(
            dregs=dregs,
            gregs=[],
            dts={
                make_ord_prefix(i)+'dreg' : dt for i,dt in zip(range(len(dregs)), dts)
            },
            modifiers=modifiers,
            operand_modifiers=operand_modifiers,
            **kwargs
        )

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[opmove_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:
        """
        move implementation/call interface
        """
        raise NotImplementedError(self.NIE_MESSAGE)


class dummy_opmove(opmove):
    """
    Dummy move operation; ISAs assign this by default to operations they do not support
    """

    def all_signatures(self) -> list[operation_signature]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[opmove_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
