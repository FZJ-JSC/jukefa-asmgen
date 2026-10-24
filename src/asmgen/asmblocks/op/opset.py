"""
Operation setting a register to what comes from the instruction stream
alone (0.6, phase 4 of docs/V06_REWORK.md): a constant (CONST), a mask
(CONSTMASK), the vector length (VLEN), indices (INDEX).

    gen.set(dreg=v, dt=adt.DOUBLE, modifiers={opset_modifier.CONST}, constant="zero")
    gen.set(dreg=v, dt=adt.SINGLE, modifiers={opset_modifier.CONST}, constant="one")   1.0 each
    gen.set(dreg=r, dt=adt.UINT64, modifiers={opset_modifier.CONST}, value=1234)
    gen.set(dreg=m, dt=adt.DOUBLE, modifiers={opset_modifier.CONSTMASK}, mask="alltrue")
    gen.set(dreg=r, modifiers={opset_modifier.VLEN})                  bytes
    gen.set(dreg=r, dt=adt.DOUBLE, modifiers={opset_modifier.VLEN})   elements
    gen.set(dreg=v, dt=adt.DOUBLE, modifiers={opset_modifier.STINDEX}, streg=r)   i * r
    gen.set(modifiers={opset_modifier.VLEN, opset_modifier.PARAM}, param="n")    %[n], inline

The register set is dreg, the operation's one output; forms with inputs
name them. A vector ISA's set carries its base ISA's for general and
floating-point registers as a member, as rvv_opmem carries riscv64_opmem.
"""
from __future__ import annotations

from abc import abstractmethod
from enum import auto
from typing import Callable

from .modifier import operation_modifier
from .constraint import minmax_constraint
from .operand import operand_shape as osh, operand_type as ot, operand_role as orl, register_type as rt
from .operand import operand_modifier
from .operation import operation
from .scalar import GREG_DTS, scalar_operands
from .signature import operation_signature
from ...registers import asm_data_type as adt, data_reg, greg_base, freg_base


class opset_modifier(operation_modifier):
    """
    What set sets
    """
    CONST = auto()      # a constant: constant="zero" or "one" (1, 1.0 in dt), or value=<int>
    CONSTMASK = auto()  # a mask: mask="alltrue" or "allfalse"
    VLEN = auto()       # the vector length in elements of dt (bytes: UINT8, the default)
    STINDEX = auto()    # indices from a stride register: element i is i * streg
    PARAM = auto()      # the destination an inline asm register operand %[param], no dreg: inline only


FREG_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF)
VREG_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF, adt.UINT64, adt.SINT64, adt.UINT32, adt.SINT32,
            adt.UINT16, adt.SINT16, adt.UINT8, adt.SINT8)


INT_DTS = (adt.UINT64, adt.SINT64, adt.UINT32, adt.SINT32, adt.UINT16, adt.SINT16, adt.UINT8, adt.SINT8)


def const_signatures(rtype, dts, constant : str = "zero") -> list[operation_signature]:
    """
    CONST constant ("zero", "one") of a register of rtype, one signature per
    type
    """
    return [operation_signature(modifiers={opset_modifier.CONST}, structural_params={"constant": constant},
                                operands={"dreg": osh(ot.REGISTER, orl.DATA, rtype, dt)}) for dt in dts]


def zero_signatures(rtype, dts) -> list[operation_signature]:
    """
    CONST zero of a register of rtype
    """
    return const_signatures(rtype, dts, "zero")


def mask_signatures(dts, rtype=rt.MASK) -> list[operation_signature]:
    """
    CONSTMASK mask="alltrue" or "allfalse" of a mask register - or a vector
    register where masks are those (AVX2) - per element type
    """
    return [operation_signature(modifiers={opset_modifier.CONSTMASK}, structural_params={"mask": mask},
                                operands={"dreg": osh(ot.REGISTER, orl.DATA, rtype, dt)})
            for mask in ("alltrue", "allfalse") for dt in dts]


def vlen_signatures() -> list[operation_signature]:
    """
    VLEN into a general register: elements of set's dt, bytes (UINT8)
    where it isn't given
    """
    return [operation_signature(modifiers={opset_modifier.VLEN}, structural_params={},
                                operands={"dreg": osh(ot.REGISTER, orl.DATA, rt.GP, adt.UINT64)})]


def vlen_param_signatures() -> list[operation_signature]:
    """
    VLEN into an inline asm register operand named param (PARAM): inline
    output only - C helpers' asm
    """
    return [operation_signature(modifiers={opset_modifier.VLEN, opset_modifier.PARAM}, structural_params={},
                                operands={"param": osh(ot.NAME, orl.PARAM, None, None)})]


class _inline_param(greg_base):
    """
    Where an inline asm register operand %[name] goes in place of a general
    register: what PARAM hands the ISAs' implementations, which write
    str(dreg)
    """
    # pylint: disable-next=super-init-not-called
    def __init__(self, name : str):
        self.name = name

    def __str__(self) -> str:
        return f"%[{self.name}]"

    @property
    def idx(self) -> int:
        return -1


def stindex_signatures(dts) -> list[operation_signature]:
    """
    STINDEX of a vector register: element i is i times the stride register
    streg (gathers' and scatters' offsets)
    """
    return [operation_signature(modifiers={opset_modifier.STINDEX}, structural_params={},
                                operands={"dreg": osh(ot.REGISTER, orl.DATA, rt.VEC, dt),
                                          "streg": osh(ot.REGISTER, orl.PARAM, rt.GP, adt.UINT64)})
            for dt in dts]


def value_signatures() -> list[operation_signature]:
    """
    CONST value=<int> of a general register: any 64-bit value, materialized
    in as many instructions as it takes
    """
    value = osh(ot.IMMEDIATE, orl.PARAM, None, None,
                value_constraints=[minmax_constraint(minval=-2**63, maxval=2**64 - 1)])
    return [operation_signature(modifiers={opset_modifier.CONST}, structural_params={},
                                operands={"dreg": osh(ot.REGISTER, orl.DATA, rt.GP, dt), "value": value})
            for dt in GREG_DTS]


def scalar_signatures(*, one_freg_dts=()) -> list[operation_signature]:
    """
    A base ISA's: zero, one and values in general registers, zero (and one,
    where the ISA has it without a scratch register) in floating-point ones
    """
    return (zero_signatures(rt.GP, GREG_DTS) + zero_signatures(rt.FP, FREG_DTS)
            + const_signatures(rt.GP, GREG_DTS, "one") + const_signatures(rt.FP, one_freg_dts, "one")
            + value_signatures())


class opset(operation):
    """
    Assembly/IR instruction(s) setting dreg from the instruction stream
    """
    NIE_MESSAGE = "Method not implemented"

    # pylint: disable-next=arguments-differ
    def __call__(self, *, dreg=None, dt : adt | None = None,
                 modifiers : set[opset_modifier] | None = None,
                 operand_modifiers : dict[str, set[operand_modifier]] | None = None,
                 **kwargs) -> str:
        """
        dreg set as the modifiers say; dt is dreg's type - but for VLEN,
        where it is the type whose elements are counted (bytes, UINT8, where
        it isn't given) and dreg a general register
        """
        modifiers = set(modifiers or ())
        dts = {"dreg": dt}
        if opset_modifier.PARAM in modifiers:
            # no dreg: an inline asm operand, by its name (param)
            dts = {}
            dt = adt.UINT8 if dt is None else dt
        elif opset_modifier.VLEN in modifiers:
            dt = adt.UINT8 if dt is None else dt
            dts = {"dreg": adt.UINT64}
        elif dt is None:
            raise ValueError(f"{type(self).__name__}: set needs dt (only VLEN counts bytes without it)")
        if dreg is not None:
            kwargs["dreg"] = dreg
        return self.execute(dregs=[], dts=dts, modifiers=modifiers,
                            operand_modifiers=operand_modifiers or {}, dt=dt, **kwargs)

    @abstractmethod
    # pylint: disable-next=arguments-differ
    def implementation(self, *, dreg, dt : adt, modifiers, operand_modifiers, **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)


class scalar_opset(opset):
    """
    set on a base ISA's general and floating-point registers: each ISA
    writes the instructions
    """
    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self.signatures = self.make_signatures()

    def make_signatures(self) -> list[operation_signature]:
        """
        The ISA's signatures (its opset/signatures.py)
        """
        return scalar_signatures()

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, dreg : data_reg, dt : adt, modifiers=None, constant=None,
                       value=None, **_) -> str:
        if value is not None:
            return self.value_greg(dreg=dreg, value=value)
        if "one" == constant:
            if isinstance(dreg, greg_base):
                return self.one_greg(dreg=dreg)
            return self.one_freg(dreg=dreg, dt=dt)
        if isinstance(dreg, greg_base):
            return self.zero_greg(dreg=dreg)
        return self.zero_freg(dreg=dreg, dt=dt)

    def one_greg(self, *, dreg : greg_base) -> str:
        """
        A general register set to 1
        """
        return self.value_greg(dreg=dreg, value=1)

    def one_freg(self, *, dreg : freg_base, dt : adt) -> str:
        """
        A floating-point register set to 1.0
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def value_greg(self, *, dreg : greg_base, value : int) -> str:
        """
        A general register set to value, in as many instructions as it takes
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def zero_greg(self, *, dreg : greg_base) -> str:
        """
        A general register zeroed
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def zero_freg(self, *, dreg : freg_base, dt : adt) -> str:
        """
        A floating-point register zeroed
        """
        raise NotImplementedError(self.NIE_MESSAGE)


class vector_opset(opset):
    """
    set on a vector ISA's registers; general and floating-point ones go to
    scalar_op, the base ISA's set
    """
    scalar_op : scalar_opset = None
    signatures : list = []

    def get_signatures(self) -> list[operation_signature]:
        return [*self.signatures, *(self.scalar_op.get_signatures() if self.scalar_op else [])]

    # pylint: disable-next=arguments-differ
    def implementation(self, *, dreg=None, dt=None, modifiers=None, operand_modifiers=None, constant=None,
                       value=None, mask=None, streg=None, param=None, **kwargs) -> str:
        if opset_modifier.PARAM in set(modifiers or ()):
            dreg = _inline_param(param)
        if opset_modifier.VLEN in set(modifiers or ()):
            return self.vlen(dreg=dreg, dt=dt)
        if opset_modifier.STINDEX in set(modifiers or ()):
            return self.stindex_reg(dreg=dreg, dt=dt, streg=streg)
        # If a scalar register is passed, forward to the base ISA's set
        if self.scalar_op is not None and isinstance(dreg, (greg_base, freg_base)):
            given = {k: v for k, v in (("constant", constant), ("value", value)) if v is not None}
            return self.scalar_op(dreg=dreg, dt=dt, modifiers=modifiers, operand_modifiers=operand_modifiers,
                                  **given, **scalar_operands(kwargs))
        if mask is not None:
            return self.mask_reg(dreg=dreg, dt=dt, mask=mask)
        if "one" == constant:
            return self.one_reg(dreg=dreg, dt=dt)
        return self.zero_reg(dreg=dreg, dt=dt)

    def mask_reg(self, *, dreg : data_reg, dt : adt, mask : str) -> str:
        """
        A mask register all true or all false for elements of dt
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def vlen(self, *, dreg : greg_base, dt : adt = adt.UINT8) -> str:
        """
        The vector length into a general register, in elements of dt (bytes:
        UINT8)
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def stindex_reg(self, *, dreg : data_reg, dt : adt, streg : greg_base) -> str:
        """
        Element i of dreg i * streg, in dt's element size
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def one_reg(self, *, dreg : data_reg, dt : adt) -> str:
        """
        Every element of a vector register 1 (1.0) in dt
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def zero_reg(self, *, dreg : data_reg, dt : adt) -> str:
        """
        A vector register (or tile) zeroed
        """
        raise NotImplementedError(self.NIE_MESSAGE)
