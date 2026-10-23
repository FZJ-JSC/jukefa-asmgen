"""
aarch64 set's signatures: CONST zero
"""
from ...op.operand import register_type as rt
from ...op.opset import FREG_DTS, zero_signatures
from ...op.scalar import GREG_DTS


def make_aarch64_opset_signatures() -> list:
    """
    CONST zero of the registers set reaches here
    """
    return zero_signatures(rt.GP, GREG_DTS) + zero_signatures(rt.FP, FREG_DTS)
