# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
data move operation for NEON
"""

from typing import Callable

from ....registers import asm_data_type as adt, data_reg, adt_size
from ...op.signature import operation_signature
from ...op.operand import operand_modifier as omod
from ...op.opmove import opmove, opmove_modifier as mod
from ...aarch64.types import aarch64_greg
from .signatures import make_neon_move_signatures
from ...aarch64.move import aarch64_move
from ...op.scalar import scalar_dregs_only, scalar_operands


class neon_move(opmove):
    """
    NEON data move instructions: mov, dup, ins
    """
    # the base ISA's moves between scalar registers: a member whose
    # signatures are offered too and to which calls with them go
    scalar_class = aarch64_move

    def __init__(self, asmwrap : Callable[[str], str],
                 dt_suffixes : dict[adt, str], dt_idxsuffixes : dict[adt, str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_idxsuffixes = dt_idxsuffixes
        self.signatures = make_neon_move_signatures()
        self.scalar_op = self.scalar_class(asmwrap=asmwrap) if self.scalar_class else None

    def all_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.all_signatures() if self.scalar_op else [])]

    def implementation(self, *, dregs : list[data_reg],
                       dts : dict[str, adt],
                       modifiers : set[mod],
                       operand_modifiers : dict[str, set[omod]],
                       **kwargs) -> str:
        # If scalar registers only are passed, forward to the base ISA's move
        if self.scalar_op is not None and scalar_dregs_only(dregs):
            return self.scalar_op(dregs=dregs, dts=[dts[k] for k in sorted(dts)],
                                  modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **scalar_operands(kwargs))

        src, dst = dregs[0], dregs[1]
        dt = dts['bdreg']
        el = self.dt_idxsuffixes[dt]
        amods = operand_modifiers.get('adreg', set())
        if {omod.ILANEROT, omod.ILANESHIFT} & amods:
            size = adt_size(dts['bdreg'])
            nbytes = ((-kwargs['adreg_irot']) % (16 // size)) * size
            return self.asmwrap(f"ext v{dst.idx}.16b,v{src.idx}.16b,v{src.idx}.16b,#{nbytes}")
        bmods = operand_modifiers.get('bdreg', set())

        if omod.ILANE in amods:
            source = f"v{src.idx}.{el}[{kwargs['adreg_lane']}]"
        elif isinstance(src, aarch64_greg):
            source = str(src.retype(dt))
        elif amods or bmods:
            # a scalar FP register is lane 0 of its vector register
            source = f"v{src.idx}.{el}[0]"
        else:
            return self.asmwrap(f"mov v{dst.idx}.16b,v{src.idx}.16b")

        if omod.BCAST in bmods:
            return self.asmwrap(f"dup v{dst.idx}.{self.dt_suffixes[dt]},{source}")
        return self.asmwrap(f"ins v{dst.idx}.{el}[{kwargs['bdreg_lane']}],{source}")
