"""
AVX set signatures: CONST zero, one
"""
from ....registers import adt_size
from ...op.constraint import minmax_constraint
from ...op.operand import register_type as rt
from ...op.opset import maskavl_signatures, INT_DTS, VREG_DTS, const_signatures, mask_signatures, stindex_signatures, vlen_signatures, zero_signatures, first_mask_signatures
from ....registers import asm_data_type as adt


def make_avx_opset_signatures(*, evex : bool = False) -> list:
    """
    zero; one in floating-point types and integers - all bits set, then
    shifted into place
    """
    # AVX-512's masks are k registers, AVX2's vector registers
    masks = mask_signatures(VREG_DTS, rtype=rt.MASK if evex else rt.VEC)
    if evex:
        # kxnorw, then kshiftrw: element types of 16 lanes or fewer in a zmm
        masks += first_mask_signatures({dt: minmax_constraint(minval=1, maxval=64 // adt_size(dt))
                                        for dt in VREG_DTS if 64 // adt_size(dt) <= 16})
        # a mask of a run-time count: all ones, bzhi to the count, kmovw - a
        # scratch general register; 16 lanes or fewer (kmovw, AVX512F)
        masks += maskavl_signatures([dt for dt in VREG_DTS if 64 // adt_size(dt) <= 16], rt.MASK, tmp=rt.GP)
    return (zero_signatures(rt.VEC, VREG_DTS)
            + const_signatures(rt.VEC, (adt.DOUBLE, adt.SINGLE, adt.HALF, *INT_DTS), "one")
            + masks
            + vlen_signatures()
            # no byte multiply: indices of 2 to 8 bytes
            + stindex_signatures((adt.DOUBLE, adt.SINGLE, adt.HALF, adt.UINT64, adt.SINT64,
                                adt.UINT32, adt.SINT32, adt.UINT16, adt.SINT16)))
