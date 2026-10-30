"""
SME set's signatures: CONST zero
"""
from ...op.operand import register_type as rt
from ...op.opset import opset_modifier, zero_signatures
from ...op.signature import operation_signature


def make_sme_opset_signatures(*, dt_suffixes : dict) -> list:
    """
    CONST zero of the registers set reaches here
    """
    return zero_signatures(rt.TILE, list(dt_suffixes)) + [
        # the engine - streaming mode and ZA - on (smstart) and off (smstop)
        operation_signature(modifiers={opset_modifier.ENGINE}, structural_params={"engine": e}, operands={})
        for e in ("on", "off")]
