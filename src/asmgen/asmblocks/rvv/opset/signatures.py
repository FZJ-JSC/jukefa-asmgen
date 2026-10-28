"""
RVV set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, VREG_DTS, const_signatures, stindex_signatures, vlen_param_signatures, vlen_signatures, zero_signatures, vlen_avl_signatures, maskavl_signatures


def make_rvv_opset_signatures() -> list:
    """
    zero in any type, one in integers (1.0 takes a scratch register)
    """
    return (zero_signatures(rt.VEC, VREG_DTS) + const_signatures(rt.VEC, INT_DTS, "one") + vlen_signatures() + vlen_param_signatures()
            + stindex_signatures(VREG_DTS)
            # vsetvli rd, avl: the length for avl elements
            + vlen_avl_signatures()
            # a mask of the first avl elements: vid.v into a scratch vector,
            # vmsltu.vx against avl (under the current element width)
            + maskavl_signatures(VREG_DTS, rt.VEC, tmp=rt.VEC))
