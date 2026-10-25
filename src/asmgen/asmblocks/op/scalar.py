"""
Scalar arithmetic and moves on general registers as operations (0.6,
phases 2 and 3 of docs/V06_REWORK.md): add, sub, mul and shift of two
registers, or of a register and an immediate in B's place (bimm) where the
ISA has the form - its range a constraint; moves between scalar registers.
A vector add, mul or move carries its base ISA's scalar operation as a
member - its signatures among its own, calls with scalar registers handed
to it - as rvv_opmem does with riscv64_opmem.

    gen.add(adreg=a, bdreg=b, cdreg=c, a_dt=UINT64, b_dt=UINT64, c_dt=UINT64)   c = a + b
    gen.add(adreg=a, bimm=8, cdreg=c, a_dt=UINT64, c_dt=UINT64)                 c = a + 8
    gen.shift(adreg=a, bimm=3, cdreg=c, a_dt=UINT64, c_dt=UINT64,
              modifiers={opd3_modifier.SHL})                                     c = a << 3
"""
from __future__ import annotations

from typing import Callable

from ...registers import asm_data_type as adt, greg_base, freg_base
from .opd3 import opd3, opd3_modifier
from .move import move, move_modifier
from .operand import operand_shape as osh, operand_type as ot, operand_role as orl, register_type as rt
from .operand import operand_modifier as opd_mod
from .signature import operation_signature

GREG_DTS = (adt.UINT64, adt.SINT64)


class scalar_arith(opd3):
    """
    An arithmetic operation on general registers: each ISA says which forms
    it has and writes the instructions (with_register, with_immediate,
    with_shifted, with_voffset)
    """
    def __init__(self, *, asmwrap : Callable[[str], str]):
        self.asmwrap = asmwrap
        self._signatures = self.make_signatures()

    def op_modifiers(self) -> list[set]:
        """
        The operation modifiers its plain forms come with (shift: SHL, SHR)
        """
        return [set()]

    def has_register_form(self) -> bool:
        """
        Whether B can be any general register (x86's shifts take only %cl)
        """
        return True

    def imm_forms(self) -> list[list]:
        """
        Forms with an immediate in B's place (bimm), one list of constraints
        per form; none: no immediate form
        """
        return []

    def voffset_forms(self) -> list[list]:
        """
        Forms whose bimm counts vector lengths (opd3_modifier.VOFFSET), one
        list of constraints per form; none: no such form (a fixed vector
        length is a byte count)
        """
        return []

    def shift_forms(self) -> list[tuple[list, frozenset]]:
        """
        Forms adding a register shifted left first (operand modifier ISHIFT
        on bdreg, the amount bdreg_shift): the amount's constraints and the
        features each needs; none: no such form
        """
        return []

    def make_signatures(self) -> list[operation_signature]:
        sigs = []
        for dt in GREG_DTS:
            def reg(dt=dt, mods=None):
                return osh(ot.REGISTER, orl.DATA, rt.GP, dt, modifiers=set(mods or ()))
            def imm(constraints):
                return osh(ot.IMMEDIATE, orl.PARAM, None, None, value_constraints=list(constraints))
            for mods in self.op_modifiers():
                if self.has_register_form():
                    sigs.append(operation_signature(modifiers=set(mods), structural_params={},
                                                    operands={"adreg": reg(), "bdreg": reg(), "cdreg": reg()}))
                for constraints in self.imm_forms():
                    sigs.append(operation_signature(modifiers=set(mods), structural_params={},
                                                    operands={"adreg": reg(), "cdreg": reg(), "bimm": imm(constraints)}))
            for constraints in self.voffset_forms():
                sigs.append(operation_signature(modifiers={opd3_modifier.VOFFSET}, structural_params={},
                                                operands={"adreg": reg(), "cdreg": reg(), "bimm": imm(constraints)}))
            for constraints, requires in self.shift_forms():
                sigs.append(operation_signature(modifiers=set(), structural_params={}, requires=frozenset(requires),
                                                operands={"adreg": reg(), "bdreg": reg(mods={opd_mod.ISHIFT}),
                                                          "cdreg": reg(), "bdreg_shift": imm(constraints)}))
        return sigs

    def all_signatures(self) -> list[operation_signature]:
        return self._signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, adreg : greg_base, cdreg : greg_base, bdreg : greg_base | None = None,
                       bimm : int | None = None, bdreg_shift : int | None = None,
                       modifiers : set | None = None, **_) -> str:
        mods = set(modifiers or ())
        if opd3_modifier.VOFFSET in mods:
            return self.with_voffset(adreg=adreg, bimm=bimm, cdreg=cdreg)
        if bdreg_shift is not None:
            return self.with_shifted(adreg=adreg, bdreg=bdreg, bdreg_shift=bdreg_shift, cdreg=cdreg)
        if bimm is not None:
            return self.with_immediate(adreg=adreg, bimm=bimm, cdreg=cdreg, modifiers=mods)
        return self.with_register(adreg=adreg, bdreg=bdreg, cdreg=cdreg, modifiers=mods)

    def with_register(self, *, adreg : greg_base, bdreg : greg_base, cdreg : greg_base,
                      modifiers : set) -> str:
        """
        c = a op b
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_immediate(self, *, adreg : greg_base, bimm : int, cdreg : greg_base,
                       modifiers : set) -> str:
        """
        c = a op bimm
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_shifted(self, *, adreg : greg_base, bdreg : greg_base, bdreg_shift : int,
                     cdreg : greg_base) -> str:
        """
        c = a op (b << bdreg_shift)
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def with_voffset(self, *, adreg : greg_base, bimm : int, cdreg : greg_base) -> str:
        """
        c = a + bimm vector lengths (in bytes)
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

    def all_signatures(self) -> list[operation_signature]:
        return self._signatures

    # pylint: disable-next=arguments-differ
    def implementation(self, *, dregs, dts, modifiers=None, **_) -> str:
        src, dst = dregs[0], dregs[1]
        if move_modifier.BITS in (modifiers or set()):
            return self.bits_to_freg(src=src, dst=dst, dt=dts["bdreg"])
        if "GP" == self.rtype_of(dts["adreg"]):
            return self.greg_to_greg(src=src, dst=dst)
        return self.freg_to_freg(src=src, dst=dst, dt=dts["bdreg"])

    @staticmethod
    def rtype_of(dt) -> str:
        return "FP" if dt in scalar_move.FP_DTS else "GP"

    def greg_to_greg(self, *, src : greg_base, dst : greg_base) -> str:
        """
        dst = src, general registers
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def freg_to_freg(self, *, src : freg_base, dst : freg_base, dt : adt) -> str:
        """
        dst = src, floating-point registers
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def bits_to_freg(self, *, src : greg_base, dst : freg_base, dt : adt) -> str:
        """
        dst's bits = src's, from a general register
        """
        raise NotImplementedError(self.NIE_MESSAGE)


def scalar_dregs_only(dregs) -> bool:
    """
    Whether a move's registers are all scalar (general or floating-point):
    its vector operation hands it to the scalar member
    """
    return all(isinstance(r, (greg_base, freg_base)) for r in dregs)

