"""
AVX set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, VREG_DTS, const_signatures, mask_signatures, stindex_signatures, vlen_signatures, zero_signatures
from ....registers import asm_data_type as adt


def make_avx_opset_signatures(*, evex : bool = False) -> list:
    """
    zero; one in floating-point types and integers - all bits set, then
    shifted into place
    """
    # AVX-512's masks are k registers, AVX2's vector registers
    masks = mask_signatures(VREG_DTS, rtype=rt.MASK if evex else rt.VEC)
    return (zero_signatures(rt.VEC, VREG_DTS)
            + const_signatures(rt.VEC, (adt.DOUBLE, adt.SINGLE, adt.HALF, *INT_DTS), "one")
            + masks
            + vlen_signatures()
            # no byte multiply: indices of 2 to 8 bytes
            + stindex_signatures((adt.DOUBLE, adt.SINGLE, adt.HALF, adt.UINT64, adt.SINT64,
                                adt.UINT32, adt.SINT32, adt.UINT16, adt.SINT16)))
