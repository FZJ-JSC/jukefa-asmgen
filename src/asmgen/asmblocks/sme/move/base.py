# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
data move operation base class for SME
"""

from typing import Callable

from ....registers import asm_data_type as adt, data_reg

from ...op.signature import operation_signature
from ...op.operand import operand_modifier
from ...op.move import move,move_modifier
from ...op.misc import make_ord_prefix as mop

from ...sve.move import sve_move

from ...sve.types import sve_vreg
from ..types import sme_treg
from ...aarch64.types import aarch64_greg,aarch64_freg

from .signatures import make_sme_move_signatures
from ...op.scalar import scalar_dregs_only, scalar_operands

class sme_move(move):
    """
    SME data move instructions
    """
    # moves between scalar registers: the SVE move's member - its
    # signatures among this one's, calls with them handed on to it
    scalar_class = None

    def __init__(self, asmwrap : Callable[[str],str],
                 dt_suffixes = dict[adt,str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes

        self.sve_move = sve_move(asmwrap=asmwrap, dt_suffixes=dt_suffixes)
        self.signatures = make_sme_move_signatures()
        self.signatures.extend(self.sve_move.all_signatures())
        self.scalar_op = self.scalar_class(asmwrap=asmwrap) if self.scalar_class else None

    def all_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.all_signatures() if self.scalar_op else [])]


    def get_v_specs(self,
                    regs: list[data_reg],
                    dt : adt) -> str:
        """
        Generate the string for accessing vector registers

        :param regs: list of vregs
        :param dt: data type
        :return: string to be used for vector register access in a mova instruction
        """

        dt_suf = self.dt_suffixes[dt]

        if len(regs) == 1:
            return f"{regs[0]}.{dt_suf}"
        return f"{{{regs[0]}.{dt_suf}-{regs[-1]}.{dt_suf}}}"

    def get_tile_spec(self,
                      *,
                      reg : sme_treg,
                      dt : adt,
                      row_or_col : operand_modifier,
                      regrc : aarch64_greg,
                      immrc : int,
                      nvregs : int) -> str:
        """
        Generate the string for accessing the tile register

        :param reg: SME tile
        :param dt: element data type
        :param row_or_col: determines V/H access type
        :param regrc: GP register containing row/column offset
        :param immrc: immediate containing row/column offset
        :param nvregs: how many rows/columns to access
        :return: string to be used for tile access in a mova instruction
        """
        # GNU as takes the slice direction in lower case: za0v.d[w12, 0]
        rc_char = "v" if operand_modifier.COL == row_or_col else "h"

        multi_spec = ""
        if nvregs > 1:
            # signatures enforce xdreg_immy are consecutive
            multi_spec = f":{immrc+nvregs-1}"

        dt_suf = self.dt_suffixes[dt]

        return f"{reg}{rc_char}.{dt_suf}[{regrc.get_wreg()},{immrc}{multi_spec}]"

    # it's fine
    # pylint: disable-next=too-many-locals
    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[move_modifier],
                       operand_modifiers : dict[str,set[operand_modifier]],
                       **kwargs) -> str:

        # If scalar registers only are passed, forward to the base ISA's move
        if self.scalar_op is not None and scalar_dregs_only(dregs):
            return self.scalar_op(dregs=dregs, dts=[dts[k] for k in sorted(dts)],
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        input_count = 1
        output_count = 1
        if move_modifier.MULTIPLE_IN in modifiers:
            input_count = kwargs['nin']
        if move_modifier.MULTIPLE_OUT in modifiers:
            output_count = kwargs['nout']

        if input_count == output_count and input_count == 1:
            if all(isinstance(reg, sve_vreg)
                   for reg in dregs) or any(isinstance(reg, (aarch64_freg,aarch64_greg))
                                            for reg in dregs):
                return self.sve_move(
                        dregs=dregs,
                        dts=[dts['adreg'],dts['bdreg']],
                        modifiers=modifiers,
                        operand_modifiers=operand_modifiers,
                        **kwargs)

        t_to_v = False
        v_to_t = False
        tile_count = 0
        if any(isinstance(reg, sme_treg) for reg in dregs[:input_count]):
            t_to_v = True
            tile_count = input_count
        if any(isinstance(reg, sme_treg) for reg in dregs[input_count:]):
            v_to_t = True
            tile_count = output_count


        tileregidx = 0
        tileregname = 'invalid'
        if t_to_v:
            tileregname = 'adreg'
        elif v_to_t:
            tileregidx = input_count
            tileregname = f"{mop(tileregidx)}dreg"

        row_or_col = {operand_modifier.ROW,
                      operand_modifier.COL}.intersection(
                              operand_modifiers[tileregname]).pop()

        regrc = 0
        immrc = 0
        if operand_modifier.ROW in operand_modifiers[tileregname]:
            regrc = kwargs[f'{tileregname}_rowreg']
            immrc = kwargs[f'{tileregname}_immrow']
        else:
            regrc = kwargs[f'{tileregname}_colreg']
            immrc = kwargs[f'{tileregname}_immcol']

        tile_spec = self.get_tile_spec(
                reg=dregs[tileregidx],
                dt=dts[tileregname],
                row_or_col=row_or_col,
                regrc=regrc,
                immrc=immrc,
                nvregs=len(dregs)-tile_count)

        vreg_specs = self.get_v_specs([r
                                       for r in dregs
                                       if not isinstance(r, sme_treg)],
                                      dt=dts[tileregname])

        predicate = f"{kwargs['amreg']}/m"

        if t_to_v:
            return self.asmwrap(
                    (f"mova {vreg_specs},"
                     f"{predicate},"
                     f"{tile_spec}"))
        if v_to_t:
            return self.asmwrap(
                    (f"mova {tile_spec},"
                     f"{predicate},"
                     f"{vreg_specs}"))

        raise ValueError("tile/vector move direction could not be determined")
