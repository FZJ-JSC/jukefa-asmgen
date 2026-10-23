"""
neon's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, data_reg
from ...aarch64.opset import aarch64_opset
from .signatures import make_neon_opset_signatures


class neon_opset(vector_opset):
    """
    NEON set: dup v.<T>,{x,w}zr; general and floating-point registers
    AArch64's
    """
    def __init__(self, *, asmwrap : Callable[[str], str], dt_suffixes : dict, dt_greg_pfx : dict):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_greg_pfx = dt_greg_pfx
        self.scalar_op = aarch64_opset(asmwrap=asmwrap)
        self.signatures = make_neon_opset_signatures(dt_suffixes=dt_suffixes, dt_greg_pfx=dt_greg_pfx)

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        return self.asmwrap(f"dup {dreg}.{self.dt_suffixes[dt]},{self.dt_greg_pfx[dt]}zr")
