"""
Scalar arithmetic on general registers as operations (0.6, phase 2 of
docs/V06_REWORK.md): add, sub and mul of two registers, or of a register
and an immediate where the ISA has the form - its range a constraint. A
vector add or mul carries its base ISA's scalar operation as a member - its
signatures among its own, calls with general registers handed to it - as
rvv_opmem does with riscv64_opmem.

    gen.add(adreg=a, bdreg=b, cdreg=c, a_dt=UINT64, b_dt=UINT64, c_dt=UINT64)   c = a + b
    gen.add(adreg=a, imm=8, cdreg=c, a_dt=UINT64, c_dt=UINT64)                  c = a + 8
"""
from __future__ import annotations

from typing import Callable

from ...registers import asm_data_type as adt
from .opd3 import opd3
from .move import move, move_modifier
from .operand import operand_shape as osh, operand_type as ot, operand_role as orl, register_type as rt
from .operand import operand_modifier as opd_mod
from .signature import operation_signature

GREG_DTS = (adt.UINT64, adt.SINT64)


class scalar_arith(opd3):
    """
    An arithmetic operation on general registers: each ISA says which
    immediate forms it has (imm_forms) and writes the instructions
    """
    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self._signatures = self.make_signatures()

    def imm_forms(self) -> list[list]:
        """
        The immediate forms' constraints, one list per form; none: no
        immediate form
        """
        return []

    def has_register_form(self) -> bool:
        """
        Whether B can be any general register (x86's shifts take only %cl)
        """
        return True

    def shift_forms(self) -> list[tuple[list, frozenset]]:
        """
        Forms adding a register shifted left first (operand modifier ISHIFT
        on bdreg, the amount bdreg_shift): the amount's constraints and the
        features each needs; none: no such form
        """
        return []

    def vl_forms(self) -> list[list]:
        """
        Forms adding an immediate count of vector lengths (operand modifier
        VLSCALE on imm), one list of constraints per form; none: no such
        form (a fixed vector length is a byte count for imm)
        """
        return []

    def make_signatures(self) -> list[operation_signature]:
        sigs = []
        for dt in GREG_DTS:
            def reg(dt=dt, mods=None):
                return osh(ot.REGISTER, orl.DATA, rt.GP, dt, modifiers=set(mods or ()))
            if self.has_register_form():
                sigs.append(operation_signature(modifiers=set(), structural_params={},
                                                operands={"adreg": reg(), "bdreg": reg(), "cdreg": reg()}))
            for constraints in self.vl_forms():
                count = osh(ot.IMMEDIATE, orl.PARAM, None, None, modifiers={opd_mod.VLSCALE},
                            value_constraints=list(constraints))
                sigs.append(operation_signature(modifiers=set(), structural_params={},
                                                operands={"adreg": reg(), "cdreg": reg(), "imm": count}))
            for constraints, requires in self.shift_forms():
                amount = osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=list(constraints))
                sigs.append(operation_signature(modifiers=set(), structural_params={}, requires=frozenset(requires),
                                                operands={"adreg": reg(), "bdreg": reg(mods={opd_mod.ISHIFT}),
                                                          "cdreg": reg(), "bdreg_shift": amount}))
            for constraints in self.imm_forms():
                imm = osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=list(constraints))
                sigs.append(operation_signature(modifiers=set(), structural_params={},
                                                operands={"adreg": reg(), "cdreg": reg(), "imm": imm}))
        return sigs

    def get_signatures(self) -> list[operation_signature]:
        return self._signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, adreg, cdreg, bdreg=None, imm=None, bdreg_shift=None,
                       operand_modifiers=None, **_) -> str:
        if imm is not None and opd_mod.VLSCALE in (operand_modifiers or {}).get("imm", set()):
            return self.with_vl(cdreg, adreg, imm)
        if bdreg_shift is not None:
            return self.with_shifted(cdreg, adreg, bdreg, bdreg_shift)
        if imm is not None:
            return self.with_immediate(cdreg, adreg, imm)
        return self.with_register(cdreg, adreg, bdreg)

    def with_register(self, c, a, b) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_immediate(self, c, a, imm : int) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_vl(self, c, a, count : int) -> str:
        """
        c = a + count vector lengths (in bytes)
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_shifted(self, c, a, b, shift : int) -> str:
        """
        c = a + (b << shift), or minus
        """
        raise NotImplementedError(self.NIE_MESSAGE)


def scalar_operands(kwargs : dict) -> dict:
    """
    What a vector operation's implementation hands on to its scalar member
    besides the named operands: anything but execute's bookkeeping
    """
    return {k: v for k, v in kwargs.items() if k not in ("dts", "dregs", "gregs")}


BITS_GP = {adt.DOUBLE: adt.UINT64, adt.SINGLE: adt.UINT32, adt.HALF: adt.UINT16}


class scalar_move(move):
    """
    Moves between scalar registers, as operations (0.6, phase 3): general
    to general (dregs [src, dst], UINT64 or SINT64), floating-point to
    floating-point, and a general register's bits into a floating-point
    register (move_modifier.BITS; UINT64 to DOUBLE, UINT32 to SINGLE,
    UINT16 to HALF). A vector move carries its base ISA's as a member, as
    rvv_opmem does with riscv64_opmem. Each ISA writes the instructions.
    """
    FP_DTS = (adt.DOUBLE, adt.SINGLE, adt.HALF)

    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self._signatures = self.make_signatures()

    def make_signatures(self) -> list[operation_signature]:
        sigs = []
        def reg(rtype, dt):
            return osh(ot.REGISTER, orl.DATA, rtype, dt)
        for dt in GREG_DTS:
            sigs.append(operation_signature(modifiers=set(), structural_params={},
                                            operands={"adreg": reg(rt.GP, dt), "bdreg": reg(rt.GP, dt)}))
        for dt in self.FP_DTS:
            sigs.append(operation_signature(modifiers=set(), structural_params={},
                                            operands={"adreg": reg(rt.FP, dt), "bdreg": reg(rt.FP, dt)}))
        for dt in self.FP_DTS:
            sigs.append(operation_signature(modifiers={move_modifier.BITS}, structural_params={},
                                            operands={"adreg": reg(rt.GP, BITS_GP[dt]), "bdreg": reg(rt.FP, dt)}))
        return sigs

    def get_signatures(self) -> list[operation_signature]:
        return self._signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, dregs, dts, modifiers=None, **_) -> str:
        src, dst = dregs[0], dregs[1]
        if move_modifier.BITS in (modifiers or set()):
            return self.bits_to_freg(src, dst, dts["bdreg"])
        if "GP" == self.rtype_of(dts["adreg"]):
            return self.greg_to_greg(src, dst)
        return self.freg_to_freg(src, dst, dts["bdreg"])

    @staticmethod
    def rtype_of(dt) -> str:
        return "FP" if dt in scalar_move.FP_DTS else "GP"

    def greg_to_greg(self, src, dst) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def freg_to_freg(self, src, dst, dt) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def bits_to_freg(self, src, dst, dt) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)


def scalar_dregs_only(dregs) -> bool:
    """
    Whether a move's registers are all scalar (general or floating-point):
    its vector operation hands it to the scalar member
    """
    from ...registers import greg_base, freg_base   # pylint: disable=import-outside-toplevel
    return all(isinstance(r, (greg_base, freg_base)) for r in dregs)

