"""
AVX set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, VREG_DTS, const_signatures, zero_signatures
from ....registers import asm_data_type as adt


def make_avx_opset_signatures() -> list:
    """
    zero; one in floating-point types and integers - all bits set, then
    shifted into place
    """
    return zero_signatures(rt.VEC, VREG_DTS) + const_signatures(rt.VEC, (adt.DOUBLE, adt.SINGLE, adt.HALF, *INT_DTS), "one")
