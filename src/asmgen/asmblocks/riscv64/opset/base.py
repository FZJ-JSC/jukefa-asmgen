"""
riscv64's set
"""
from ...op.opset import scalar_opset
from ....registers import asm_data_type as adt, greg_base, freg_base
from ..types import FCDT_SUFFIXES
from .signatures import make_riscv64_opset_signatures


class riscv64_opset(scalar_opset):
    """
    RISC-V set: add r, zero, 0; fcvt.{d,s,h}.w f, zero
    """
    def make_signatures(self):
        return make_riscv64_opset_signatures()

    def zero_greg(self, *, dreg : greg_base) -> str:
        return self.asmwrap(f"add {dreg}, zero, 0")

    def zero_freg(self, *, dreg : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fcvt.{FCDT_SUFFIXES[dt]}.w {dreg}, zero")

    def value_greg(self, *, dreg : greg_base, value : int) -> str:
        return self.asmwrap(f"li {dreg},{value}")
