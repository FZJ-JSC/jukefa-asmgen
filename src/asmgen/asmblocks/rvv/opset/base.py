"""
rvv's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, data_reg
from ...riscv64.opset import riscv64_opset
from .signatures import make_rvv_opset_signatures


class rvv_opset(vector_opset):
    """
    RVV set: vmv.v.i v,0; general and floating-point registers RISC-V's
    """
    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self.scalar_op = riscv64_opset(asmwrap=asmwrap)
        self.signatures = make_rvv_opset_signatures()

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        return self.asmwrap(f"vmv.v.i {dreg},0")
