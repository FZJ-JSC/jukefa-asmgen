"""
RVV set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, VREG_DTS, const_signatures, vlen_signatures, zero_signatures


def make_rvv_opset_signatures() -> list:
    """
    zero in any type, one in integers (1.0 takes a scratch register)
    """
    return zero_signatures(rt.VEC, VREG_DTS) + const_signatures(rt.VEC, INT_DTS, "one") + vlen_signatures()
