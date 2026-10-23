"""
AVX set's signatures: CONST zero
"""
from ...op.operand import register_type as rt
from ...op.opset import VREG_DTS, zero_signatures


def make_avx_opset_signatures() -> list:
    """
    CONST zero of the registers set reaches here
    """
    return zero_signatures(rt.VEC, VREG_DTS)
