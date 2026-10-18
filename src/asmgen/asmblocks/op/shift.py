"""
Signatures of moves shifting or rotating a register's lanes or blocks
"""

from .operand import (
    operand_modifier as omod,
    operand_shape as osh,
    operand_type as ot,
    operand_role as orl,
    register_type as rgt,
)
from .constraint import minmax_constraint, otherplusn_constraint
from .signature import operation_signature as opsig


def shift_signatures(dts, forms, irot, fill=None, destructive=False, blocksize=None) -> list:
    """
    adreg the source, bdreg the destination, adreg_irot how far (lanes or
    blocks; negative towards lane 0), fillreg the value FILLOTHERS fills in.

    :param forms: the source's modifier sets
    :param irot: (dt, modifiers) -> (min, max) of irot, None where not offered
    :param fill: dt -> register type of fillreg
    :param destructive: the destination must be the source
    :param blocksize: dt -> elements per block, for the block forms
    """
    sigs = []
    for dt in dts:
        for amods in forms:
            bounds = irot(dt, amods)
            if bounds is None:
                continue
            operands = {
                "adreg": osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
                "bdreg": osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
                "adreg_irot": osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
                    minmax_constraint(minval=bounds[0], maxval=bounds[1])]),
            }
            if omod.FILLOTHERS in amods:
                operands["fillreg"] = osh(otype=ot.REGISTER, orole=orl.DATA, rtype=fill(dt), dt=dt)
            if destructive:
                operands["bdreg"].value_constraints = [otherplusn_constraint(
                    other="adreg", offset=0, getint=lambda r: r.idx, what="register")]
            operands["adreg"].modifiers = set(amods)
            params = {}
            if {omod.IBLOCKROT, omod.IBLOCKSHIFT} & set(amods):
                params["adreg_blocksize"] = blocksize(dt)
            sigs.append(opsig(modifiers=set(), structural_params=params, operands=operands))
    return sigs


def fill_signatures(dts, irot, scalar) -> list:
    """
    A scalar into a vector shifted in place: adreg the scalar, bdreg the
    vector - read and written - shifted by bdreg_irot, the scalar filling the
    lane vacated (ILANESHIFT + FILLOTHERS on bdreg). A fill loop's step.

    :param irot: dt -> irot, +1 or -1
    :param scalar: dt -> register type of the scalar
    """
    sigs = []
    for dt in dts:
        r = irot(dt)
        operands = {
            "adreg": osh(otype=ot.REGISTER, orole=orl.DATA, rtype=scalar(dt), dt=dt),
            "bdreg": osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
            "bdreg_irot": osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
                minmax_constraint(minval=r, maxval=r)]),
        }
        operands["bdreg"].modifiers = {omod.ILANESHIFT, omod.FILLOTHERS}
        sigs.append(opsig(modifiers=set(), structural_params={}, operands=operands))
    return sigs


def extract_signatures(dts, scalar) -> list:
    """
    Lane 0 of a vector into a scalar register: adreg the vector (ILANE, lane
    0), bdreg the scalar. A drain loop's step
    """
    sigs = []
    for dt in dts:
        operands = {
            "adreg": osh(otype=ot.REGISTER, orole=orl.DATA, rtype=rgt.VEC, dt=dt),
            "bdreg": osh(otype=ot.REGISTER, orole=orl.DATA, rtype=scalar(dt), dt=dt),
            "adreg_lane": osh(otype=ot.IMMEDIATE, orole=orl.PARAM, value_constraints=[
                minmax_constraint(minval=0, maxval=0)]),
        }
        operands["adreg"].modifiers = {omod.ILANE}
        sigs.append(opsig(modifiers=set(), structural_params={}, operands=operands))
    return sigs
