# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
RVV 0.7.1's loads and stores (0.6, phase 5): RVV's operations, 0.7.1's
mnemonics. An element width is one letter of the mnemonic - vle.v loads
SEW-wide elements (64 bits here), vlw.v, vlh.v and vlb.v 32, 16 and 8 bits,
sign-extended to SEW - so a load or store is right only at the width the
kernel's vsetvli set (loads at another one, 1.0's mixed precision, aren't a
goal). Strides and broadcasts are vls*.v, indices vlx*.v (vsux*.v stores,
unordered), segments vlseg<nf>*.v; indices are SEW-wide, as wide as the
data.
"""
from __future__ import annotations

from ...registers import adt_size, ait_size
from ...registers import asm_data_type as adt
from ..op import opmem_modifier as mod
from ..op.operand import operand_modifier as opd_mod
from ..op.signature import operation_signature
from ..rvv.opmem import rvv_load, rvv_store

#: the element width's letter in 0.7.1's mnemonics, by its bytes
WIDTH_LETTER = {8: "e", 4: "w", 2: "h", 1: "b"}


class rvv071_opmem_forms:
    """
    0.7.1's mnemonics and index width over RVV's loads and stores
    """

    def get_instruction(self, base : str, modifiers : set[mod],
                        operand_modifiers : dict[str, set[opd_mod]],
                        dt : adt, **kwargs) -> str:
        has_bcast = any(opd_mod.BCAST in mods for mods in operand_modifiers.values())
        inst = base
        if mod.GSTRIDE in modifiers or has_bcast:
            inst += "s"
        if mod.VINDEX in modifiers:
            inst += "x" if base.startswith("vl") else "ux"
        if mod.STRUCT in modifiers:
            inst += f"seg{kwargs['nstructs']}"
        return inst + WIDTH_LETTER[adt_size(dt)] + ".v"

    def all_signatures(self) -> list[operation_signature]:
        # 0.7.1's indices are SEW-wide: as wide as the data
        return [s for s in super().all_signatures()
                if mod.VINDEX not in s.modifiers
                or ait_size((s.structural_params or {})["it"]) == adt_size(s.operands["adreg"].dt)]


class rvv071_load(rvv071_opmem_forms, rvv_load):
    """
    RVV 0.7.1 vector loads
    """


class rvv071_store(rvv071_opmem_forms, rvv_store):
    """
    RVV 0.7.1 vector stores
    """
