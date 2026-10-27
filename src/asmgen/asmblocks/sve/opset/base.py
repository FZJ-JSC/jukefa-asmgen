"""
sve's set
"""
from typing import Callable

from ...op.opset import vector_opset
from ....registers import asm_data_type as adt, adt_size, data_reg, greg_base
from ...aarch64.opset import aarch64_opset
from .signatures import make_sve_opset_signatures


class sve_opset(vector_opset):
    """
    SVE set: dup z.<T>,#0; general and floating-point registers AArch64's
    """
    def __init__(self, *, asmwrap : Callable[[str], str], dt_suffixes : dict):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.scalar_op = aarch64_opset(asmwrap=asmwrap)
        self.signatures = make_sve_opset_signatures(dt_suffixes=dt_suffixes)

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        return self.asmwrap(f"dup {dreg}.{self.dt_suffixes[dt]},#0")

    def one_reg(self, *, dreg : data_reg, dt : adt) -> str:
        if dt in (adt.DOUBLE, adt.SINGLE, adt.HALF):
            return self.asmwrap(f"fdup {dreg}.{self.dt_suffixes[dt]},#1.0")
        return self.asmwrap(f"dup {dreg}.{self.dt_suffixes[dt]},#1")

    def mask_reg(self, *, dreg : data_reg, dt : adt, mask : str, count : int | None = None) -> str:
        if "first" == mask:
            if dreg.is_pn:
                raise ValueError(f"{dreg}: a predicate-as-counter takes no ptrue pattern")
            return self.asmwrap(f"ptrue {dreg}.{self.dt_suffixes[dt]}, vl{count}")
        if "alltrue" == mask:
            return self.asmwrap(f"ptrue {dreg}.{self.dt_suffixes[dt]}")
        # pfalse is .b only; a predicate-as-counter register is its p's
        return self.asmwrap(f"pfalse p{dreg.idx}.b")

    def maskavl_reg(self, *, dreg : data_reg, dt : adt, avlreg : greg_base) -> str:
        if dreg.is_pn:
            raise ValueError(f"{dreg}: a predicate-as-counter's whilelo takes a vector count (vlx2): not yet")
        return self.asmwrap(f"whilelo {dreg}.{self.dt_suffixes[dt]}, xzr, {avlreg}")

    def vlen(self, *, dreg : greg_base, dt : adt = adt.UINT8) -> str:
        suf = {1: "b", 2: "h", 4: "w", 8: "d"}[adt_size(dt)]
        return self.asmwrap(f"cnt{suf} {dreg}")

    def stindex_reg(self, *, dreg : data_reg, dt : adt, streg : greg_base) -> str:
        # w for elements below 8 bytes
        step = streg.get_wreg() if adt_size(dt) < 8 else streg
        return self.asmwrap(f"index {dreg}.{self.dt_suffixes[dt]},#0,{step}")
