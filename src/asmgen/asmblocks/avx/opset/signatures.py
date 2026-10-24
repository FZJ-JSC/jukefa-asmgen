"""
AVX set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, VREG_DTS, const_signatures, mask_signatures, vlen_signatures, zero_signatures
from ....registers import asm_data_type as adt


def make_avx_opset_signatures(*, evex : bool = False) -> list:
    """
    zero; one in floating-point types and integers - all bits set, then
    shifted into place
    """
    masks = mask_signatures(VREG_DTS)
    if not evex:
        # AVX2's masks are vector registers (gathers' and scatters')
        masks += mask_signatures(VREG_DTS, rtype=rt.VEC)
    return (zero_signatures(rt.VEC, VREG_DTS)
            + const_signatures(rt.VEC, (adt.DOUBLE, adt.SINGLE, adt.HALF, *INT_DTS), "one")
            + masks
            + vlen_signatures())
