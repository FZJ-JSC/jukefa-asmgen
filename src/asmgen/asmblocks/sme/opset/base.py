"""
sme's set
"""
from typing import Callable

from ....registers import asm_data_type as adt, data_reg, treg_base
from ...sve.opset import sve_opset
from .signatures import make_sme_opset_signatures


class sme_opset(sve_opset):
    """
    SME set: SVE's, and zero {za<n>.<T>} for a tile
    """
    def __init__(self, *, asmwrap : Callable[[str], str], dt_suffixes : dict):
        super().__init__(asmwrap=asmwrap, dt_suffixes=dt_suffixes)
        self.signatures = self.signatures + make_sme_opset_signatures(dt_suffixes=dt_suffixes)

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        if isinstance(dreg, treg_base):
            # the tiles to zero are a list: zero {za0.d}
            return self.asmwrap(f"zero {{{dreg}.{self.dt_suffixes[dt]}}}")
        return super().zero_reg(dreg=dreg, dt=dt)
