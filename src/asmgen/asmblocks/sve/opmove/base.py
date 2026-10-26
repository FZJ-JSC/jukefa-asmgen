# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
data move operation base class for SVE
"""

from ....registers import freg_base
from typing import Callable

from ....registers import adt_size, asm_data_type as adt, data_reg

from ...op.signature import operation_signature
from ...op.operand import operand_modifier as omod
from ...op.opmove import opmove,opmove_modifier as mod

from ..types import sve_vreg
from ...aarch64.types import aarch64_greg,aarch64_freg


from .signatures import make_sve_move_signatures
from ...aarch64.opmove import aarch64_move
from ...op.scalar import scalar_dregs_only, scalar_operands

class sve_move(opmove):
    """
    SVE data move instructions
    """
    # the base ISA's moves between scalar registers: a member whose
    # signatures are offered too and to which calls with them go
    scalar_class = aarch64_move

    def __init__(self, asmwrap : Callable[[str],str],
                 dt_suffixes = dict[adt,str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes

        self.signatures = make_sve_move_signatures()
        self.scalar_op = self.scalar_class(asmwrap=asmwrap) if self.scalar_class else None

    def all_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.all_signatures() if self.scalar_op else [])]

    def build_vec_lane_spec(self,
                            vreg : sve_vreg,
                            lane : int,
                            dt : adt) -> str:
        """
        Generate specification for accessing a specific lane of an SVE vreg

        :param vreg: register to access
        :param lane: lane to access
        :param dt: element data type
        :return: string that can be used as operand to an SVE instruction
        """

        dt_suf = self.dt_suffixes[dt]
        return f"{vreg}.{dt_suf}[{lane}]"


    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[mod],
                       operand_modifiers : dict[str,set[omod]],
                       **kwargs) -> str:

        # If scalar registers only are passed, forward to the base ISA's move
        if self.scalar_op is not None and scalar_dregs_only(dregs):
            return self.scalar_op(dregs=dregs, dts=[dts[k] for k in sorted(dts)],
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        predicate = ""
        if mod.MASK in modifiers:
            predicate = f"{kwargs['amreg']}/m,"

        input_reg = dregs[0]
        output_reg = dregs[1]

        input_dt = dts['adreg']
        output_dt = dts['bdreg']

        input_mods = operand_modifiers.get('adreg',set())
        output_mods = operand_modifiers.get('bdreg', set())
        suffixes = {8: "d", 4: "s", 2: "h", 1: "b"}
        if omod.FILLOTHERS in output_mods:
            return self.asmwrap(f"insr {output_reg}.{suffixes[adt_size(output_dt)]},{input_reg}")
        if omod.ILANE in input_mods and isinstance(output_reg, freg_base):
            # lane 0 is the register's low part
            return self.asmwrap(f"fmov {output_reg},{suffixes[adt_size(input_dt)]}{input_reg.idx}")
        if {omod.ILANEROT, omod.ILANESHIFT, omod.IBLOCKROT} & input_mods:
            size = adt_size(output_dt)
            if omod.FILLOTHERS in input_mods:
                suffix = {8: "d", 4: "s", 2: "h", 1: "b"}[size]
                return self.asmwrap(f"insr {output_reg}.{suffix},{kwargs['fillreg']}")
            unit = 16 if omod.IBLOCKROT in input_mods else size
            return self.asmwrap(f"ext {output_reg}.b,{output_reg}.b,{input_reg}.b,"
                                f"#{-kwargs['adreg_irot'] * unit}")
        # not used for making decision rn
        #output_mods = operand_modifiers.get('bdreg',set())

        if omod.IBLOCK in input_mods:
            return self.asmwrap(f"dup {output_reg}.q,{input_reg}.q[{kwargs['adreg_block']}]")
        inst = "mov"
        in_str = ""
        out_str = f"{output_reg}.{self.dt_suffixes[output_dt]}"
        if omod.ILANE in input_mods:
            in_str = self.build_vec_lane_spec(input_reg, kwargs['adreg_lane'], input_dt)
            inst = "dup"
        elif isinstance(input_reg, (aarch64_freg,aarch64_greg)):
            in_str = str(input_reg.retype(input_dt))
        else:
            in_str = f"{input_reg}.{self.dt_suffixes[input_dt]}"

        return self.asmwrap(f"{inst} {out_str},{predicate}{in_str}")
