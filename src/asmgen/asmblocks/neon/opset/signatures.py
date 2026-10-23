"""
neon set's signatures: CONST zero
"""
from ...op.operand import register_type as rt
from ...op.opset import zero_signatures


def make_neon_opset_signatures(*, dt_suffixes : dict, dt_greg_pfx : dict) -> list:
    """
    CONST zero of the registers set reaches here
    """
    return zero_signatures(rt.VEC, [dt for dt in dt_suffixes if dt in dt_greg_pfx])
