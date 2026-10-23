"""
NEON set signatures: CONST zero, one
"""
from ...op.operand import register_type as rt
from ...op.opset import const_signatures, zero_signatures
from ....registers import asm_data_type as adt

ONE_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF, adt.UINT32, adt.SINT32, adt.UINT16, adt.SINT16, adt.UINT8, adt.SINT8)


def make_neon_opset_signatures(*, dt_suffixes : dict, dt_greg_pfx : dict) -> list:
    """
    zero; one in floating-point types (fmov #1.0) and integers up to 32 bits
    (movi #1)
    """
    return (zero_signatures(rt.VEC, [dt for dt in dt_suffixes if dt in dt_greg_pfx])
            + const_signatures(rt.VEC, [dt for dt in ONE_DTS if dt in dt_suffixes], "one"))
