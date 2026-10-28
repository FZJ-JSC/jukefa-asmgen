"""
What an ISA needs set up before a kernel uses its vector or matrix
instructions, and torn down after: governing predicates, an engine mode, the
vector length, constants in memory.

A generator's isa_setup() returns the object for its ISA. Each declares its
steps - what kind, what for, whether every kernel needs it - so a code
generator can tell what it is getting, rather than calling a black box. The
old isaquirks(), isaendquirks() and isadata() remain, as wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto



class setup_kind(Enum):
    #: governing predicates for masked instructions (SVE, SME)
    PREDICATE = auto()
    #: a mode the vector or matrix unit must be put in (SME streaming mode, ZA)
    ENGINE = auto()
    #: the vector length, set in a register (RVV)
    VECTOR_LENGTH = auto()
    #: data instructions read from memory (AVX gather and scatter indices)
    CONSTANTS = auto()


class predication(Enum):
    """
    How vector instructions limit the elements they work on
    """
    #: they don't: fixed-width vectors (NEON, AVX2)
    NONE = auto()
    #: instructions name a predicate register (SVE, SME): an all-true one for
    #: full vectors, others for partial ones - the last vector of an edge
    EXPLICIT = auto()
    #: the vector length is state every instruction obeys (RVV's vl)
    IMPLICIT = auto()


@dataclass(frozen=True)
class setup_step:
    """
    :param required: needed by every kernel using the ISA's vector
                     instructions; otherwise only when an instruction asks for it
    """
    kind : setup_kind
    description : str
    required : bool = True


class isa_setup:
    """
    Nothing to set up: NEON, AVX2. The base of the others.
    """

    def __init__(self, gen):
        self.gen = gen

    def state(self, dt) -> dict[str, str]:
        """
        The processor state the setup leaves (signatures' state_requires and
        state_writes name the same states) - none by default
        """
        return {}

    def teardown_state(self) -> dict[str, str]:
        """the processor state the teardown leaves - none by default"""
        return {}

    def steps(self) -> list[setup_step]:
        return []

    def predication(self) -> predication:
        return predication.NONE

    def supports_vl_override(self) -> bool:
        """
        Whether setup() can run vectors shorter than the hardware's: on RVV the
        effective vector length is set in a register and everything uses it.
        SVE has no equivalent - MUL VL offsets and incd count full vectors.
        """
        return False

    def element_type_is_state(self) -> bool:
        """
        Whether the element type is state the instructions run under (RVV's
        vtype, set by vsetvli) rather than part of each instruction (SVE,
        NEON, AVX): where it is, an instruction on another element type - or
        on registers grouped otherwise - needs setup_dt first
        """
        return False

    def setup_dt(self, *, dt, group : int = 1) -> str:
        """
        The element type state switched to dt over registers grouped by
        group - a wide accumulator of a widening instruction spans group
        registers - keeping the vector length. Nothing where the element type
        is part of each instruction. Whatever the ISA groups registers by
        otherwise (RVV's LMUL) is its own: group counts the registers the
        caller sees
        """
        return ""

    def clobbers(self) -> dict[str, list[int]]:
        """
        Registers the setup and teardown clobber beyond those they allocate,
        for the calling convention to keep where they are the callee's
        """
        return {}

    def setup(self, *, rt, dt, vl : int | None = None) -> str:
        """
        Instructions setting up everything every kernel needs, in order

        :param vl: elements per vector, fewer than the hardware's, where
                   supports_vl_override(); None for the hardware's
        """
        if vl is not None and not self.supports_vl_override():
            raise ValueError(f"{type(self.gen).__name__} cannot override the vector length")
        return ""

    def teardown(self, *, rt, dt) -> str:
        return ""

    def data(self) -> str:
        """
        Constants instructions have registered, in a read-only section of their
        own: written inside an inline asm statement they would otherwise sit in
        the instruction stream
        """
        asmdata = self.gen.asmdata
        if not asmdata:
            return ""
        wrap = self.gen.asmwrap
        result = wrap(".pushsection .rodata")
        for name, datalist in asmdata.items():
            result += wrap(f".balign {self.gen.simd_size}")
            result += wrap(f"{self.gen.labelstr(name)}:")
            for d in datalist:
                result += wrap(f"  {str(d)}")
        result += wrap(".popsection")
        return result

    def all_true_predicate(self, *, rt, dt):
        """
        The predicate register masked instructions use for full vectors, set
        up by setup(); None where instructions aren't masked
        """
        return None

    def vector_length_register(self, *, rt):
        """
        The register holding the vector length in elements, set by setup();
        None where it isn't kept in one
        """
        return None
