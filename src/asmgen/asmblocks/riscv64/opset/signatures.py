"""
riscv64's set signatures: CONST zero, one, values
"""
from ...op.opset import scalar_signatures


def make_riscv64_opset_signatures() -> list:
    """
    zero in general and floating-point registers, one in general ones (1.0 takes a scratch register), values in general ones
    """
    return scalar_signatures()
