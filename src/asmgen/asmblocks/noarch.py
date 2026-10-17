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


from enum import Enum,auto
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

    @abstractmethod
    def get_parameters(self) -> list[str]:
        """
        Returns a list of parameter names that this generator supports

        :return: List of parameter names
        :rtype: list[str]
        """
        raise NotImplementedError(NIE_MESSAGE)

    def get_param_value(self, name : str) -> int|str:
        """
        Returns the current value of the named generator parameter

        :param name: Name of the parameter
        :type name: str
        :return: Current value of the parameter
        :rtype: int|str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def set_parameter(self, name : str, value : int|str):
        """
        Sets a parameter in the generator, affecting the output, min/max values
        and supported features

        :param name: Name of the parameter
        :type name: str
        :param value: New value to set the parameter to
        :type value: int|str
        """
        raise NotImplementedError(NIE_MESSAGE)

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
        Returns the SIMD size in bytes or 1 if the ISA is vector-length-agnostic

        :return: SIMD size in bytes or 1 for VLA ISAs
        :rtype: 1
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def c_simd_size_function(self) -> str:
        """
        Returns a string that contains a c function definition for 
        determining the SIMD size in bytes at runtime

        :return: string containing a c function called get_simd_size()
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def simd_size_to_greg(self, *, reg : greg_type,
                          dt : asm_data_type) -> str:
        """
        Returns a string containing instructions for writing the SIMD size
        in elements to a GP register

        :param reg: GP register to write simd size to
        :type reg: greg_type
        :param dt: Element data type
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: string containing the instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

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

    @property
    @abstractmethod
    def are_fregs_in_vregs(self) -> bool:
        """
        Returns True if for the underlying ISA scalar registers are contained 
        inside vector registers, which is for example the case with ARM NEON and SVE

        :return : True if scalar regs are inside vector regs, otherwise False
        :rtype: bool
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def indexable_elements(self, dt : asm_data_type) -> int:
        """
        Returns the number of directly indexable elements in a SIMD/vector register

        :param dt: Data type to check
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: Number of directly indexable elements
        :rtype: int
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

    @property
    @abstractmethod
    def min_prefetch_offset(self) -> int:
        """
        Returns the minimum immediate prefetch offset in bytes (can be negative)

        :return: Minimum immediate prefetch offset
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_prefetch_offset(self) -> int:
        """
        Returns the maximum immediate prefetch offset in bytes (can be negative)

        :return: Maximum immediate prefetch offset
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def min_load_voff(self) -> int:
        """
        Returns the minimum immediate offset in number of SIMD/vector registers
        for loading data into SIMD/vector registers  (can be negative)

        :return: Minimum immediate vector loading offset in vectors
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_load_voff(self) -> int:
        """
        Returns the maximum immediate offset in number of SIMD/vector registers
        for loading data into SIMD/vector registers (can be negative)

        :return: Maximum immediate vector loading offset in vectors
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_add_voff(self) -> int:
        """
        Returns the maximum immediate value in number of SIMD/vector registers
        for adding vector-length-based byte-size-offsets to a GP register

        :return: Maximum immediate value
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def min_load_immoff(self, dt : asm_data_type) -> int:
        """
        Returns the minimum immediate offset in bytes
        for loading data into SIMD/vector registers  (can be negative)

        :return: Minimum immediate vector loading offset in bytes
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def max_load_immoff(self, dt : asm_data_type) -> int:
        """
        Returns the maximum immediate offset in bytes
        for loading data into SIMD/vector registers  (can be negative)

        :return: Maximum immediate vector loading offset in bytes
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def min_bcast_immoff(self, dt : asm_data_type) -> int:
        """
        Returns the minimum immediate offset in bytes
        for broadcasting a value from memory into SIMD/vector
        registers  (can be negative)

        :return: Minimum immediate vector broadcasting offset in bytes
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def max_bcast_immoff(self, dt : asm_data_type) -> int:
        """
        Returns the maximum immediate offset in bytes
        for broadcasting a value from memory into SIMD/vector
        registers  (can be negative)

        :return: Maximum immediate vector broadcasting offset in bytes
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def min_fload_immoff(self, dt : asm_data_type) -> int:
        """
        Returns the minimum immediate offset in bytes
        for loading data into scalar registers  (can be negative)

        :return: Minimum immediate scalar loading offset in bytes
        :rtype: int
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def max_fload_immoff(self, dt : asm_data_type) -> int:
        """
        Returns the maximum immediate offset in bytes
        for loading data into scalar registers  (can be negative)

        :return: Maximum immediate scalar loading offset in bytes
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
    def cb(self, *, reg1 : greg_type, reg2 : greg_type, cmp : comparison, label : str) -> str:
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
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_greg(self, *, greg : greg_type) -> str:
        """
        Returns the string containing instructions to set a GP register to zero
        
        :param greg: GP register to set to zero
        :type greg: class:`asmgen.register.greg_base`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_freg(self, *, freg : freg_type, dt : asm_data_type) -> str:
        """
        Returns the string containing instructions to set a scalar register to zero
        
        :param freg: scalar register to set to zero
        :type freg: class:`asmgen.register.freg_base`
        :param dt: Data type contained in the scalar register
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_vreg(self, *, vreg : vreg_type, dt : asm_data_type) -> str:
        """
        Returns the string containing instructions to set all elements in a 
        vector register to zero
        
        :param vreg: vector register to set to zero
        :type vreg: class:`asmgen.register.vreg_base`
        :param dt: Data type contained in the scalar register
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_treg(self, *, treg : treg_type, dt : asm_data_type) -> str:
        """
        Returns the string containing instructions to set all elements in a 
        tile register to zero
        
        :param treg: tile register to set to zero
        :type treg: class:`asmgen.register.vreg_base`
        :param dt: Data type contained in the scalar register
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_greg(self, *, src : greg_type, dst : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to copy the content of one
        GP register to another GP register

        :param src: GP register to copy data from
        :type src: class:`asmgen.register.greg_base`
        :param dst: GP register to copy data to
        :type dst: class:`asmgen.register.greg_base`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_greg(self, *, areg : greg_type, offset : int, dst : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to load data from the address in
        one GP register into another GP register

        :param areg: GP register containing the data address
        :type areg: class:`asmgen.register.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param dst: GP register to load data into
        :type dst: class:`asmgen.register.greg_base`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_greg(self, *, areg : greg_type, offset : int, src : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to store data from
        one GP register to the memory address contained in another GP register

        :param areg: GP register containing the data address
        :type areg: class:`asmgen.register.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param src: GP register to store data from
        :type src: class:`asmgen.register.greg_base`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_freg(self, *, src : freg_type, dst : freg_type, dt : asm_data_type) -> str:
        """
        Returns the string containing the instruction(s) to copy the content of one
        scalar register to another scalar register

        :param src: scalar register to copy data from
        :type src: class:`asmgen.register.freg_base`
        :param dst: scalar register to copy data to
        :type dst: class:`asmgen.register.freg_base`
        :param dt: Data type contained in the scalar register
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_greg_to_param(self, *, src : greg_type, param : str) -> str:
        """
        Returns the string containing the instruction(s) to copy the content of a
        general purpose register to an output operand (inline ASM only)

        :param src: GP register to copy data from
        :type src: class:`asmgen.register.greg_base`
        :param param: inline ASM output operand to copy data to
        :type param: str
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_param_to_greg(self, *, param : str, dst : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to copy the content of an
        input operand to a general purpose register (inline ASM only)

        :param param: inline ASM input operand to copy data from
        :type param: str
        :param dst: GP register to copy data to
        :type dst: class:`asmgen.register.greg_base`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_param_to_greg_shift(self, *, param : str, dst : greg_type,
                                bit_count : int) -> str:
        """
        Returns the string containing the instruction(s) to copy the content of an
        input operand to a general purpose register (inline ASM only), shifting it
        by the specified number of places (left = positive, right = negative)

        :param param: inline ASM input operand to copy data from
        :type param: str
        :param dst: GP register to copy data to
        :type dst: class:`asmgen.register.greg_base`
        :param bit_count: number of bits to shift the data to the left
        :type bit_count: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_greg_imm(self, *, reg : greg_type, imm : int) -> str:
        """
        Returns the string containing the instruction(s) to copy an immediate
        to a general purpose register

        :param reg: GP register to put the value in
        :type reg: class:`asmgen.register.greg_base`
        :param imm: integer value to put into the GP register
        :type imm: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mul_greg_imm(self, *, src : greg_type, dst : greg_type, factor : int) -> str:
        """
        Returns the string containing the instruction(s) to write the product
        of a GP register and an immediate into a GP register

        :param src: GP register containing one of the multiplicands
        :type src: class:`asmgen.register.greg_base`
        :param dst: GP register to write the result to
        :type dst: class:`asmgen.register.greg_base`
        :param factor: integer value to multiply the value in the src register with
        :type factor: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mul_greg_greg(self, *, dst : greg_type, reg1 : greg_type, reg2 : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to write the product of 2 GP registers
        into a GP register
        
        :param dst: destination GP register
        :type dst: greg_type
        :param reg1: first multiplicand
        :type reg1: greg_type
        :param reg2: second multiplicand
        :type reg2: greg_type
        :return: ASM/IR of the operation
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def add_greg_imm(self, *, reg : greg_type, imm : int) -> str:
        """
        Returns the string containing the instruction(s) to write the sum
        of a GP register and an immediate into a GP register

        :param src: GP register containing one of the summands
        :type src: class:`asmgen.register.greg_base`
        :param dst: GP register to write the result to
        :type dst: class:`asmgen.register.greg_base`
        :param imm: integer value to add to the value in the src register
        :type imm: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def add_greg_voff(self, *, reg : greg_type, offset : int,
                      dt : asm_data_type) -> str:
        """
        Returns the string containing the instruction(s) to write the sum
        of a GP register and an immediate multiplied by the simd/vector size in bytes
        into itself

        :param dst: GP register to write the result to
        :type dst: class:`asmgen.register.greg_base`
        :param factor: integer value to multiply with the simd/vector size in bytes and 
            add to the initial value in the GP register to
        :type factor: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def add_greg_greg(self, *, dst : greg_type, reg1 : greg_type, reg2 : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to write the sum of 2 GP registers
        into a GP register
        
        :param dst: destination GP register
        :type dst: greg_type
        :param reg1: first addend
        :type reg1: greg_type
        :param reg2: second addend
        :type reg2: greg_type
        :return: ASM/IR of the operation
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def sub_greg_greg(self, *, dst : greg_type, reg1 : greg_type, reg2 : greg_type) -> str:
        """
        Returns the string containing the instruction(s) to write the difference of 2 GP registers
        into a GP register
        
        :param dst: destination GP register
        :type dst: greg_type
        :param reg1: minuend
        :type reg1: greg_type
        :param reg2: subtrahend
        :type reg2: greg_type
        :return: ASM/IR of the operation
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def shift_greg_left(self, *, reg : greg_type, bit_count : int) -> str:
        """
        Returns the string containing the instruction(s) to left-shift the value
        in a GP register by the specified number of places

        :param reg: GP register to modify
        :type reg: class:`asmgen.register.greg_base`
        :param bit_count: number of bits to shift the value to the left
        :type bit_count: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def shift_greg_right(self, *, reg : greg_type, bit_count : int) -> str:
        """
        Returns the string containing the instruction(s) to right-shift the value
        in a GP register by the specified number of places

        :param reg: GP register to modify
        :type reg: class:`asmgen.register.greg_base`
        :param bit_count: number of bits to shift the value to the right
        :type bit_count: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def greg_to_voffs(self, *, streg : greg_type, vreg : vreg_type, dt : asm_data_type) -> str:
        """
        Returns the string containing the instruction(s) to create indices
        for a gather/scatter instruction with the value in the GP register as stride

        :param reg: GP register containing the stride
        :type reg: class:`asmgen.register.greg_base`
        :param reg: Vector register to write the indices to
        :type reg: class:`asmgen.register.vreg_base`
        :param dt: Data type of the value
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def prefetch_l1_immoff(self, *, areg : greg_type, offset : int):
        """
        Returns the string containing the instruction(s) to issue a prefetch 
        targeting the L1 cache

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.register.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_pointer(self, *, areg : greg_type, name : str):
        """
        Returns the string containing the instruction(s) to load a pointer from an
        input operand into a general purpose register (inline ASM only)

        :param name: inline ASM input operand to copy data from
        :type name: str
        :param areg: GP register to load the address into
        :type areg: class:`asmgen.register.greg_base`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_scalar_immoff(self, *, areg : greg_type, offset : int,
                           freg : freg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load a scalar value
        into a scalar register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param freg: scalar register to load the value into
        :type freg: class:`asmgen.registers.freg_base`
        :param dt: Data type of the value
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_scalar_immoff(self, *, areg : greg_type, offset : int,
                            freg : freg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store the value of a scalar
        register into memory

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param freg: scalar register containing the value to store
        :type freg: class:`asmgen.registers.freg_base`
        :param dt: Data type of the value
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_freg(self, *, areg : greg_type, offset : int,
                  dst : freg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load a scalar value
        into a scalar register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param freg: scalar register to load the value into
        :type freg: class:`asmgen.registers.freg_base`
        :param dt: Data type of the value
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_freg(self, *, areg : greg_type, offset : int,
                   src : freg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store the value of a scalar
        register into memory

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param freg: scalar register containing the value to store
        :type freg: class:`asmgen.registers.freg_base`
        :param dt: Data type of the value
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_voff(self, *, areg : greg_type, voffset : int,
                         vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load contiguous elements
        into a vector register, with an immediate offset being given in numbers of
        vectors

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param voffset: immediate offset in number of vectors
        :type voffset: int
        :param vreg: vector register to load the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_immoff(self, *, areg : greg_type, offset : int,
                           vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load contiguous elements
        into a vector register, with an immediate offset being given in numbers of
        bytes

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param vreg: vector register to load the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector(self, *, areg : greg_type, vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load contiguous elements
        into a vector register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to load the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def fill_vector(self, *, sreg : freg_type, vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to fill every element of a
        vector register with the data from the FP register

        :param sreg: FP register containing the data to use
        :type sreg: class:`asmgen.registers.freg_base`
        :param vreg: vector register to fill
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_lane(self, *, areg : greg_base,
                         vreg : vreg_base, lane : int, dt : asm_data_type) -> str:
        """
        Returns the string containing the instruction(s) to load a scalar value
        into the specified lane of the vector register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to load the value into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param lane: vector lane to lead the value into
        :type lane: int
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_immstride(self, *, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load elements
        into a vector register, with the byte-stride between elements being given
        by an immediate

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param byte_stride: stride between elements in bytes
        :type byte_stride: int
        :param vreg: vector register to load the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_gregstride(self, *, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load elements
        into a vector register, with the byte-stride between elements being given
        by a GP register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param sreg: GP register containing the stride between elements in bytes
        :type sreg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to load the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_gather(self, *, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, dt : asm_data_type,
                           it : asm_index_type):
        """
        Returns the string containing the instruction(s) to load elements
        into a vector register, with the offsets for each element being
        given by the element in another vector register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offvreg: vector register containing the offsets in bytes for each element
        :type offvreg: class:`asmgen.registers.vreg_base`
        :param vreg: vector register to load the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :param it: Index type of the offsets
        :type it: class:`asmgen.registers.asm_index_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_bcast1(self, *, areg : greg_type,
                          vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to broadcast a single
        scalar value from memory into all vector register lanes

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to broadcast the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_bcast1_immoff(self, *, areg : greg_type, offset : int,
                               vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to broadcast a single
        scalar value from memory into all vector register lanes, with the memory
        address being given by a base address in a GP register and an immediate
        offset in bytes

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param vreg: vector register to broadcast the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_bcast1_inc(self, *, areg : greg_type, offset : int,
                              vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to broadcast a single
        scalar value from memory into all vector register lanes, with the memory
        address being given by a base address in a GP register, which is incremented
        by an immediate offset in bytes after the load

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes to increment the value in the
            GP register by after the load
        :type offset: int
        :param vreg: vector register to broadcast the values into
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector(self, *, areg : greg_type,
                     vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store contiguous elements
        from a vector register into memory

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to store the values from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_lane(self, *, areg : greg_base,
                          vreg : vreg_base, lane : int, dt : asm_data_type) -> str:
        """
        Returns the string containing the instruction(s) to save a scalar value
        from the specified lane of the vector register into memory

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to take the value from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param lane: vector lane containing the value
        :type lane: int
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_voff(self, *, areg : greg_type, voffset : int,
                          vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store contiguous elements
        from a vector register into memory, with an immediate offset being given in 
        number of vectors

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param voffset: immediate offset in number of vectors
        :type voffset: int
        :param vreg: vector register to store the values from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_immoff(self, *, areg : greg_type, offset : int,
                           vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store contiguous elements
        from a vector register, with an immediate offset being given in number of
        bytes

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offset: immediate offset in bytes
        :type offset: int
        :param vreg: vector register to store the values from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_immstride(self, *, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store elements
        from a vector register into memory, with the byte-stride between 
        elements being given by an immediate

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param byte_stride: stride between elements in bytes
        :type byte_stride: int
        :param vreg: vector register to store the values from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_gregstride(self, *, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store elements
        from a vector register into memory, with the byte-stride between 
        elements being given by a GP register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param sreg: GP register containing the stride between elements in bytes
        :type sreg: class:`asmgen.registers.greg_base`
        :param vreg: vector register to store the values from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_scatter(self, *, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, dt : asm_data_type,
                             it : asm_index_type):
        """
        Returns the string containing the instruction(s) to store elements
        from a vector register into memory, with the offsets for each element being
        given by the element in another vector register

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param offvreg: vector register containing the offsets in bytes for each element
        :type offvreg: class:`asmgen.registers.vreg_base`
        :param vreg: vector register to store the values from
        :type vreg: class:`asmgen.registers.vreg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :param it: Index type of the offsets
        :type it: class:`asmgen.registers.asm_index_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_tile(self, *, areg : greg_type,
                  treg : treg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to load elements
        into a tile register from memory

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param treg: tile register to load the values into
        :type treg: class:`asmgen.registers.treg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_tile(self, *, areg : greg_type,
                   treg : treg_type, dt : asm_data_type):
        """
        Returns the string containing the instruction(s) to store elements
        from a tile register into memory

        :param areg: GP register containing the base address
        :type areg: class:`asmgen.registers.greg_base`
        :param treg: tile register to store the values from
        :type treg: class:`asmgen.registers.treg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the required ASM instructions
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)
