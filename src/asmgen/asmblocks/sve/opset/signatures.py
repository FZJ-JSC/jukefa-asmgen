"""
SVE set's signatures: CONST zero
"""
from ...op.operand import register_type as rt
from ...op.opset import zero_signatures


def make_sve_opset_signatures(*, dt_suffixes : dict) -> list:
    """
    CONST zero of the registers set reaches here
    """
    return zero_signatures(rt.VEC, list(dt_suffixes))
