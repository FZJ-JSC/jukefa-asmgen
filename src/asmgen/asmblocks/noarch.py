# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base ASM generator class and ISA-independent utilities
"""

# Might restructure asmgen further in the future, similarly to
# what was done with opd3, disable for now
# pylint: disable=too-many-lines


from .op.operation import operation
from enum import Enum,auto
import re
from abc import ABC, abstractmethod
from typing import TypeAlias,Union,TYPE_CHECKING

from ..registers import (
    reg_tracker,
    asm_data_type,
    asm_index_type,
    data_reg,
    greg_base, freg_base, vreg_base, treg_base
)
from ..asmdata import asm_data

from ..util import NIE_MESSAGE

if TYPE_CHECKING:
    from ..callconv.callconv import callconv

class comparison(Enum):
    """
    Types of comparisons
    """
    NZ = auto() # non-zero
    EZ = auto() # equal-to-zero
    NE = auto() # not-equal
    EQ = auto() # equal
    LE = auto() # less or equal
    GE = auto() # greater or equal
    LT = auto() # less than
    GT = auto() # greater than

class asmgen(ABC):
    """
    Abstract interface for asm code generator
    """
    greg_type : TypeAlias = greg_base
    freg_type : TypeAlias = freg_base
    vreg_type : TypeAlias = vreg_base
    treg_type : TypeAlias = treg_base

    def __init__(self):
        """
        Constructor method
        """
        self.output_inline = True

        self.asmdata : dict[str,list[asm_data]] = dict()

    @abstractmethod
    def create_callconv(self, name : str) -> "callconv":
        """
        Creates a callconv object that can be used to deal with register 
        saving/restoring/function parameters in ASM
        :param name: Name of the specific calling convention or "default"
        :type name: str
        :return: Calling convention object
        :rtype: class:`asmgen.callconv.callconv.callconv`
        """

    #: optional extensions this ISA's signatures may require: ISA parameters,
    #: "on" or "off"
    FEATURES : tuple[str, ...] = ()
    #: encoding features - signatures may require them as they do extensions -
    #: and whether they're on by default: ISA parameters, "on" or "off", that
    #: set_features doesn't touch (x86's enable-disp8 and enable-disp32)
    ENCODING_FEATURES : dict[str, bool] = {}

    def available_features(self) -> list[str]:
        """
        The optional extensions signatures of this ISA may require
        """
        return list(self.FEATURES)

    def set_features(self, names):
        """
        Enable exactly these optional extensions: signatures requiring others
        are hidden from then on (without a call, nothing is hidden)
        """
        unknown = set(names) - set(self.FEATURES)
        if unknown:
            raise ValueError(f"{type(self).__name__} has no features {sorted(unknown)}: "
                             f"{list(self.FEATURES)}")
        self.features = set(names)
        self._gate_features()

    def _encoding_features(self) -> dict[str, bool]:
        """
        This generator's encoding features, on or off: their defaults until set
        """
        if "_encoding" not in self.__dict__:
            self.__dict__["_encoding"] = dict(self.ENCODING_FEATURES)
        return self.__dict__["_encoding"]

    def _gate_features(self):
        """
        Hide from every operation the signatures requiring a feature that's
        off: the extensions set_features didn't enable (none without a call)
        and the encoding features that are off
        """
        features = getattr(self, "features", None)
        if features is None and not self.ENCODING_FEATURES:
            return
        extensions = set(self.FEATURES) if features is None else set(features)
        encodings = {name for name, on in self._encoding_features().items() if on}
        enabled = frozenset(extensions | encodings)
        for value in vars(self).values():
            if isinstance(value, operation):
                value.enabled_features = enabled

    def get_parameters(self) -> list[str]:
        """
        The names of this generator's parameters: its features here - ISA
        extensions and encoding features, "on" or "off" - backends with
        parameters of their own adding them to these

        :return: List of parameter names
        :rtype: list[str]
        """
        return list(self.FEATURES) + list(self.ENCODING_FEATURES)


    def get_param_value(self, name : str) -> int|str:
        """
        Returns the current value of the named generator parameter: a
        feature's here, "on" or "off"; backends with parameters of their own
        answer for those and ask this for the rest

        :param name: Name of the parameter
        :type name: str
        :return: Current value of the parameter
        :rtype: int|str
        """
        if "dummy" == name:
            return "dummy"
        if name in self.FEATURES:
            return "on" if name in getattr(self, "features", set()) else "off"
        if name in self.ENCODING_FEATURES:
            return "on" if self._encoding_features()[name] else "off"
        raise ValueError(f"Invalid isa parameter \"{name}\"")


    def set_parameter(self, name : str, value : int|str):
        """
        Sets a parameter in the generator, affecting the output, min/max values
        and supported features: a feature here, "on" or "off"; backends with
        parameters of their own set those and pass the rest on to this

        :param name: Name of the parameter
        :type name: str
        :param value: New value to set the parameter to
        :type value: int|str
        """
        on = str(value).lower() in ("on", "1", "yes", "true")
        if name in self.ENCODING_FEATURES:
            self._encoding_features()[name] = on
            self._gate_features()
            return
        if name in self.FEATURES:
            features = set(getattr(self, "features", set()))
            (features.add if on else features.discard)(name)
            self.set_features(sorted(features))
            return
        raise ValueError(f"Invalid name {name} or value {value}")


    def set_output_inline(self, yesno : bool):
        """
        Sets internal state that determines whether the generator emits normal ASM
        or inline ASM, i.e  "instruction\\n\\t"

        :param yesno: Whether the generator should output inline ASM
        :type yesno: bool
        """
        self.output_inline = yesno

    def labelstr(self, label : str):
        """
        Return modified label string for inline or normal ASM use
        :param label: label name
        :type label: str
        """

        if self.output_inline:
            return f".{label}%="
        return f".{label}"

    def enabled_features(self) -> set[str]:
        """
        The ISA extensions this generator may use: those set_features
        enabled, every one of FEATURES without a call
        """
        features = getattr(self, "features", None)
        return set(self.FEATURES) if features is None else set(features)

    def compiler_flags(self, compiler : str = "gcc") -> list[str]:
        """
        The flags a C compiler needs to compile and assemble this generator's
        output: its ISA and the extensions it may use (enabled_features) -
        set_features([]) first for the baseline's alone

        :param compiler: the compiler's name or path - gcc, g++, clang,
            clang++, cross-prefixed or not; armclang where it targets the ISA
        :return: the flags
        """
        raise NotImplementedError(NIE_MESSAGE)

    #: one nop: its bytes and encoding, for align's offsets
    NOP_BYTES : int = 0
    NOP_ENCODING : int = 0
    #: a shorter nop, (bytes, encoding), for what's left of an offset
    SHORT_NOP : tuple[int, int] | None = None

    def nop(self) -> str:
        """
        One nop
        """
        return self.asmwrap("nop")

    def align(self, bytes : int, offset : int = 0) -> str:   # pylint: disable=redefined-builtin
        """
        What follows aligned to bytes, offset bytes past the boundary: .balign
        - the assembler pads code with nops - then offset bytes of nops,
        written as their encodings so the assembler can't compress them.
        (Zeros, which .space writes, aren't safe to execute: a loop head's
        padding runs.)

        :param bytes: the alignment, a power of 2
        :param offset: bytes past the boundary: nops of NOP_BYTES, then one
            SHORT_NOP for what's left, if the ISA has one
        """
        if bytes < 1 or bytes & (bytes - 1):
            raise ValueError(f"alignment {bytes} isn't a power of 2")
        if not 0 <= offset < bytes:
            raise ValueError(f"offset {offset} isn't within the alignment {bytes}")
        text = self.asmwrap(f".balign {bytes}")
        if offset:
            count, rest = divmod(offset, self.NOP_BYTES) if self.NOP_BYTES else (0, offset)
            if not self.NOP_BYTES or (rest and (self.SHORT_NOP is None or rest != self.SHORT_NOP[0])):
                raise ValueError(f"{type(self).__name__}'s nops are {self.NOP_BYTES} bytes: "
                                 f"no offset of {offset}")
            if count:
                text += self.asmwrap(f".fill {count}, {self.NOP_BYTES}, {self.NOP_ENCODING:#x}")
            if rest:
                text += self.asmwrap(f".fill 1, {self.SHORT_NOP[0]}, {self.SHORT_NOP[1]:#x}")
        return text

    def asmfile_annotations(self) -> str:
        """
        What an assembly file says once, whatever functions it holds: that it
        needs no executable stack (GNU linkers assume one without the note).
        Nothing in inline asm, whose compiler writes it.
        """
        if self.output_inline:
            return ""
        return self.asmwrap('.section .note.GNU-stack,"",%progbits')

    def asmwrap(self, code : str) -> str:
        """
        Wrap the ASM instruction according according to output_inline

        :param code: string to wrap
        :type: str
        :return: Wrapped instruction string
        :rtype: str
        """
        lines = code.strip().split('\n')
        if self.output_inline:
            return "".join(f"\"{line}\\n\\t\"\n" for line in lines)
        return "".join(f"{line}\n" for line in lines)

    @staticmethod
    def operands(inputs : list[tuple[str,str,str]],
                 outputs : list[tuple[str,str,str]],
                 clobber : list[Union[greg_base|data_reg]]) -> str:
        """
        Returns the closing block of an inline asm block, containing the inputs,
        outputs and clobber list

        :param inputs: list of tuples containing input operand specifications in the form of
            ("asmname", "constraint", "c expression")
        :type outputs: list[tuple[str,str,str]]
        :param outputs: list of tuples containing output operand specifications in the form of
            ("asmname", "constraint", "c expression")
        :type inputs: list[tuple[str,str,str]]
        :param clobber: list of clobbered registers
        :type clobber: list[Union[greg_base|data_reg]]
        :return: string containing the block to be inserted at the end of an inline asm
            statement in a c/c++ source
        :rtype: str
        """
        opblock  = ": "
        opblock += ",".join([f"[{n}] \"{t}\" ({init})" for n,t,init in outputs])
        opblock += "\n"
        opblock += ": "
        opblock += ",".join([f"[{n}] \"{t}\" ({init})" for n,t,init in inputs])
        opblock += "\n"
        opblock += ": "
        opblock += ",".join([f"\"{reg}\"" for reg in clobber])
        return opblock

    def reserved_registers(self) -> dict[str, dict[str, int]]:
        """
        Registers no kernel may allocate - stack pointer, link register and the
        like: register type -> name -> index
        """
        return {}

    def shared_registers(self) -> list[tuple[str, str, dict[int, int] | None]]:
        """
        Register types that are the same registers: (type, other type, index
        mapping or None for the same indices)
        """
        return []

    @property
    def fadd(self):
        """
        Deprecated (0.6): add - the operation adds integer registers too
        """
        import warnings
        warnings.warn("asmgen: fadd is deprecated, use add", DeprecationWarning, stacklevel=2)
        return self.add

    @property
    def fmul(self):
        """
        Deprecated (0.6): mul - the operation multiplies integer registers too
        """
        import warnings
        warnings.warn("asmgen: fmul is deprecated, use mul", DeprecationWarning, stacklevel=2)
        return self.mul

    def get_reg_tracker(self) -> reg_tracker:
        """
        A register tracker for this ISA: every register file with its size,
        the reserved registers reserved and aliased by their names
        """
        def count(name):
            return getattr(self, name)()
        types = [("greg", count("max_gregs")), ("vreg", count("max_vregs"))]
        if count("max_fregs"):
            types.append(("freg", count("max_fregs")))
        if count("max_mregs"):
            types.append(("mreg", count("max_mregs")))
        rt = reg_tracker(types)
        tracked = {t for t, _ in types}
        for type_a, type_b, mapping in self.shared_registers():
            if type_a in tracked and type_b in tracked:
                rt.share(type_a, type_b, mapping)
        for tag, regs in self.reserved_registers().items():
            for name, idx in regs.items():
                rt.reserve_specific_reg(tag, idx)
                rt.alias_reg(tag, name, idx)
                # reserved means never handed out, not overwritten: a
                # function's boilerplate saves only what the code clobbers
                rt.clobbered_regs[tag].discard(idx)
        return rt

    def isa_setup(self):
        """
        What this ISA needs set up before a kernel and torn down after: see
        :class:`asmgen.asmblocks.setup.isa_setup`. Nothing, by default.
        """
        from .setup import isa_setup
        return isa_setup(self)

    def isaquirks(self, *, rt : reg_tracker, dt : asm_data_type) -> str:
        """
        Instructions setting up what every kernel needs - the predicate of
        SVE, SME's smstart, RVV's vsetvli. Wraps isa_setup().setup().
        """
        return self.isa_setup().setup(rt=rt, dt=dt)

    def isaendquirks(self, *, rt : reg_tracker, dt : asm_data_type) -> str:
        """
        Instructions tearing it down - SME's smstop. Wraps isa_setup().teardown().
        """
        return self.isa_setup().teardown(rt=rt, dt=dt)

    def isadata(self) -> str:
        """
        Constants instructions have registered, like AVX-512's gather indices,
        in a read-only section of their own. Wraps isa_setup().data().
        """
        return self.isa_setup().data()

    def isaclear(self):
        """
        Clears the internal isa data storage

        """
        self.asmdata.clear()

    @abstractmethod
    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        """
        Given a cpuinfo string, return whether this generator is compatible with the described CPU
        
        :param cpuinfo: string containing a description of the CPU, i.e. /proc/cpuinfo
        :type cpuinfo: str
        :return: True if the generator is supported, otherwise False
        :rtype: bool
        """
        raise NotImplementedError(NIE_MESSAGE)

    def supported_on_host(self) -> bool:
        """
        Checks whether this generator is supported on the current machine
        
        :return: True if the generator is supported, otherwise False
        :rtype: bool
        """
        with open("/proc/cpuinfo","r", encoding="utf-8") as f:
            cpuinfo = f.read()
        return self.supportedby_cpuinfo(cpuinfo)

    @abstractmethod
    def greg(self, reg_idx : int) -> greg_base:
        """
        Given a register index, returns the object describing the respective
        general purpose register for use as an argument to other methods

        :param reg_idx: Integer index of the general purpose register
        :type reg_idx: int
        :return: object suitable for use in methods that accept a general purpose register
        :rtype: class:`asmgen.registers.greg_base`
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def freg(self, reg_idx : int, dt : asm_data_type) -> freg_base:
        """
        Given a register index, returns the object describing the respective
        scalar register for use as an argument to other methods

        :param reg_idx: Integer index of the scalar register
        :type reg_idx: int
        :param dt: Data type that is/will be contained in the register
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: object suitable for use in methods that accept a scalar register
        :rtype: class:`asmgen.registers.freg_base`
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def vreg(self, reg_idx : int) -> vreg_base:
        """
        Given a register index, returns the object describing the respective
        vector register for use as an argument to other methods

        :param reg_idx: Integer index of the vector register
        :type reg_idx: int
        :return: object suitable for use in methods that accept a vector register
        :rtype: class:`asmgen.registers.vreg_base`
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def treg(self, reg_idx : int, dt : asm_data_type) -> treg_base:
        """
        Given a register index, returns the object describing the respective
        tile register for use as an argument to other methods

        :param reg_idx: Integer index of the tile register
        :type reg_idx: int
        :param dt: Data type that is/will be contained in the register
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: object suitable for use in methods that accept a tile register
        :rtype: class:`asmgen.registers.treg_base`
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def simd_size(self) -> int:
        """
        The SIMD size in bytes, known at compile time - raising on
        vector-length-agnostic ISAs, whose length is the machine's (set with
        VLEN gives it at run time; is_vla tells them apart)

        :return: SIMD size in bytes
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    def c_simd_size_function(self) -> str:
        """
        A C function, get_simd_size(), returning the vector length in bytes:
        the constant where it is fixed, else set(VLEN) in inline asm
        """
        if not self.is_vla:
            return f"static inline size_t get_simd_size() {{ return {self.simd_size}; }}"
        pre = self.output_inline
        self.set_output_inline(yesno=True)
        try:
            asm = self._c_simd_size_asm()
        finally:
            self.set_output_inline(yesno=pre)
        result  = "static inline size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "".join("        " + line + "\n" for line in asm.splitlines() if line.strip())
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        return result

    def _c_simd_size_asm(self) -> str:
        """
        c_simd_size_function's asm: set(VLEN) into the inline asm operand
        byte_size (PARAM)
        """
        from .op.opset import opset_modifier   # pylint: disable=import-outside-toplevel
        return self.set(modifiers={opset_modifier.VLEN, opset_modifier.PARAM}, param="byte_size")

    @abstractmethod
    def kiterkleft(self, *, kreg : greg_type,
                   kleftreg : greg_type,
                   tmpreg : greg_type,
                   unroll : int) -> str:
        """
        calculates number of unrolled iterations and tail iterations and 
        places the results in the specified GP registers

        kleft = k % unroll
        k = k // unroll

        :param kreg: GP register containing the number of iterations. will be
                     overwritten with the number of unrolled iterations
        :type kreg: greg_type
        :param kleftreg: GP register to write the number of tail iterations to
        :type kleftreg: greg_type
        :param unroll: unroll factor
        :type unroll: greg_type
        :return: string containing the ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def is_vla(self) -> bool:
        """
        Returns True if the ISA is vector-length-agnostic and False if the ISA is fixed

        :return : True if the ISA is VLA, otherwise False
        :rtype: bool
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def max_tregs(self, dt : asm_data_type) -> int:
        """
        Returns the number of available tile registers for a given data type

        :param dt: Data type to check
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: Number of available tile registers
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_vregs(self) -> int:
        """
        Returns the number of available SIMD/vector registers

        :return: Number of available SIMD/vector registers
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_fregs(self) -> int:
        """
        Returns the number of available scalar registers

        :return: Number of available scalar registers
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_gregs(self) -> int:
        """
        Returns the number of available general purpose registers

        :return: Number of available general purpose registers
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_mregs(self) -> int:
        """
        Returns the number of available mask registers

        :return: Number of available mask registers
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def label(self, *, label : str) -> str:
        """
        Returns the string containing the named ASM label

        :param label: Label name
        :type label: str
        :return: String containing correctly placed ASM label
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    def ret(self) -> str:
        """
        Return from a function: after the boilerplate's restoring block
        """
        return self.asmwrap("ret")

    @abstractmethod
    def jump(self, *, label : str) -> str:
        """
        Returns the string containing instructions for an unconditional 
        jump to the named label

        :param label: Label name
        :type label: str
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jzero(self, *, reg : greg_type, label : str) -> str:
        """
        Returns the string containing instructions for a conditional 
        jump to the named label if the GP register contains 0

        :param reg: GP register to test
        :type reg: class:`asmgen.registers.greg_base`
        :param label: Label name
        :type label: str
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jfzero(self, *, freg1 : freg_type, freg2 : freg_type,
               greg : greg_type, label : str,
               dt : asm_data_type) -> str:
        """
        Returns the string containing instructions for a conditional 
        jump to the named label if the scalar register contains 0

        :param freg1: scalar register to test
        :type freg1: class:`asmgen.registers.freg_base`
        :param freg2: additional scalar register to use if the ISA requires it
        :type freg2: class:`asmgen.registers.freg_base`
        :param greg: additional GP register to use if the ISA requires it
        :type greg: class:`asmgen.registers.greg_base`
        :param label: Label name
        :type label: str
        :param dt: Data type to test
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jvzero(self, *, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type, greg : greg_type, label : str,
               dt : asm_data_type) -> str:
        """
        Returns the string containing instructions for a conditional 
        jump to the named label if the vector register contains all 0s

        :param vreg1: vector register to test
        :type vreg1: class:`asmgen.registers.vreg_base`
        :param vreg2: additional vector register to use if the ISA requires it
        :type vreg2: class:`asmgen.registers.vreg_base`
        :param freg: additional scalar register to use if the ISA requires it
        :type freg: class:`asmgen.registers.freg_base`
        :param greg: additional GP register to use if the ISA requires it
        :type greg: class:`asmgen.registers.greg_base`
        :param label: Label name
        :type label: str
        :param dt: Data type to test
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def loopbegin(self, *, reg : greg_type, label : str) -> str:
        """
        Returns the string containing instructions for the start of a loop
        using the label as the loop start and the register as loop counter

        :param reg: GP register to use as loop counter
        :type reg: class:`asmgen.register.greg_base`
        :param label: Label name
        :type label: str
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def loopbegin_nz(self, *, reg : greg_type, label : str, labelskip : str) -> str:
        """
        Returns the string containing instructions for the start of a loop
        using the label as the loop start and the register as loop counter.
        Additionally, check the register for zero before the first iteration
        and jump to a different label if the condition is true.

        :param reg: GP register to use as loop counter
        :type reg: class:`asmgen.register.greg_base`
        :param label: Label name
        :type label: str
        :param labelskip: Label name to jump to if the GP register contains zero
        :type labelskip: str
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def loopend(self, *, reg : greg_type, label : str) -> str:
        """
        Returns the string containing instructions for the end of a loop
        using the label as the loop start and the register as loop counter

        :param reg: GP register to use as loop counter
        :type reg: class:`asmgen.register.greg_base`
        :param label: Label name
        :type label: str
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def cb(self, *, reg1, reg2, cmp : comparison, label : str,
           dt : asm_data_type | None = None, greg : greg_type | None = None) -> str:
        """
        Returns the string containing instructions to branch conditionally

        :param reg1: first GP register to compare
        :type reg1: class:`asmgen.register.greg_base`
        :param reg2: second GP register to compare
        :type reg2: class:`asmgen.register.greg_base`
        :param comparison: How the register is compared (in case of nz or ez)
        :type comparison: str
        :param label: Label name
        :type label: str
        :return: String containing the required ASM instructions
        :rtype: str
                reg1 and reg2 may be FP registers of data type dt: compared as floating
        point (EQ/EZ, NE/NZ), into greg where the ISA needs a general register
        (RISC-V); reg2 None compares with 0.0 where the ISA can (AArch64)
        """
        raise NotImplementedError(NIE_MESSAGE)

def compiler_family(compiler : str) -> str:
    """
    A C compiler's family by its name or path: "gnu" (gcc, g++), "llvm"
    (clang, clang++), "arm" (armclang) - cross prefixes and versions allowed
    """
    name = compiler.replace("\\", "/").rsplit("/", 1)[-1]
    if "armclang" in name:
        return "arm"
    if "clang" in name:
        return "llvm"
    if re.search(r"(^|-)(gcc|g\+\+|cc|c\+\+)(-\d+(\.\d+)*)?$", name):
        return "gnu"
    raise ValueError(f"{compiler}: no known compiler (gcc, g++, clang, clang++, armclang)")
