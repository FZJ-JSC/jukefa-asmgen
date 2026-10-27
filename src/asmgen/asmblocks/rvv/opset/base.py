"""
rvv's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, adt_size, data_reg, greg_base
from ...riscv64.opset import riscv64_opset
from .signatures import make_rvv_opset_signatures


class rvv_opset(vector_opset):
    """
    RVV set: vmv.v.i v,0; general and floating-point registers RISC-V's
    """
    def __init__(self, *, asmwrap : Callable[[str], str], lmul_getter : Callable[[], int] = lambda: 1,
                 policy : bool = True):
        self.asmwrap = asmwrap
        self.lmul_getter = lmul_getter
        self.policy = policy
        self.scalar_op = riscv64_opset(asmwrap=asmwrap)
        self.signatures = make_rvv_opset_signatures()

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        return self.asmwrap(f"vmv.v.i {dreg},0")

    def one_reg(self, *, dreg : data_reg, dt : adt) -> str:
        return self.asmwrap(f"vmv.v.i {dreg},1")

    def vlen(self, *, dreg : greg_base, dt : adt = adt.UINT8, avlreg : greg_base | None = None) -> str:
        # vsetvli's VLMAX: a register group's elements (bytes: e8) - or, an
        # avl given, the length for its elements, at most VLMAX; it sets the
        # vector type too, and RVV 0.7.1's has no tail and mask policy
        bits = 8 * adt_size(dt)
        policy = ", ta, ma" if self.policy else ""
        avl = "zero" if avlreg is None else str(avlreg)
        return self.asmwrap(f"vsetvli {dreg}, {avl}, e{bits}, m{self.lmul_getter()}{policy}")

    def stindex_reg(self, *, dreg : data_reg, dt : adt, streg : greg_base) -> str:
        return self.asmwrap(f"vid.v {dreg}") + self.asmwrap(f"vmul.vx {dreg},{dreg},{streg}")
