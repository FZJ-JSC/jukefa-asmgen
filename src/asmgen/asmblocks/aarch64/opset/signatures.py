"""
aarch64's set signatures: CONST zero, one, values
"""
from ...op.opset import FREG_DTS, scalar_signatures


def make_aarch64_opset_signatures() -> list:
    """
    zero in general and floating-point registers, one (1.0 too), values in general ones
    """
    return scalar_signatures(one_freg_dts=FREG_DTS)
