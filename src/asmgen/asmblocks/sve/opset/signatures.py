"""
SVE set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import INT_DTS, const_signatures, zero_signatures
from ....registers import asm_data_type as adt

ONE_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF, *INT_DTS)


def make_sve_opset_signatures(*, dt_suffixes : dict) -> list:
    """
    zero; one in floating-point types (fdup #1.0) and integers (dup #1)
    """
    return (zero_signatures(rt.VEC, list(dt_suffixes))
            + const_signatures(rt.VEC, [dt for dt in ONE_DTS if dt in dt_suffixes], "one"))
