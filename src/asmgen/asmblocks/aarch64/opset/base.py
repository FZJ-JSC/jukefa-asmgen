"""
aarch64's set
"""
from ...op.opset import scalar_opset
from ....registers import asm_data_type as adt, adt_size, greg_base, freg_base
from .signatures import make_aarch64_opset_signatures


class aarch64_opset(scalar_opset):
    """
    AArch64 set: mov x,#0; fmov from xzr or wzr
    """
    def make_signatures(self):
        return make_aarch64_opset_signatures()

    def zero_greg(self, *, dreg : greg_base) -> str:
        return self.asmwrap(f"mov {dreg},#0")

    def zero_freg(self, *, dreg : freg_base, dt : adt) -> str:
        return self.asmwrap(f"fmov {dreg},{'xzr' if 8 == adt_size(dt) else 'wzr'}")
