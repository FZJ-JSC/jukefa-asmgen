"""
SVE set signatures: CONST zero, one
"""
from ...op.constraint import oneof_constraint
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, const_signatures, mask_signatures, stindex_signatures, vlen_param_signatures, vlen_signatures, zero_signatures, first_mask_signatures
from ....registers import asm_data_type as adt

ONE_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF, *INT_DTS)


#: the counts ptrue's patterns name (vl1-vl8, vl16 ... vl256): elements
#: active where the vector has as many, none where it has fewer
SVE_VL_PATTERNS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256)


def make_sve_opset_signatures(*, dt_suffixes : dict) -> list:
    """
    zero; one in floating-point types (fdup #1.0) and integers (dup #1);
    predicates all true or all false
    """
    return (zero_signatures(rt.VEC, list(dt_suffixes))
            + const_signatures(rt.VEC, [dt for dt in ONE_DTS if dt in dt_suffixes], "one")
            + mask_signatures(list(dt_suffixes))
            # ptrue p.T, vlN: N one of the patterns'
            + first_mask_signatures({dt: oneof_constraint(valset=set(SVE_VL_PATTERNS))
                                     for dt in dt_suffixes})
            + vlen_signatures() + vlen_param_signatures()
            + stindex_signatures(list(dt_suffixes)))
