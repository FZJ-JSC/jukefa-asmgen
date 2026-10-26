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

from ..registers import asm_data_type as adt
from .op.opset import opset_modifier


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
        asmdata = getattr(self.gen, "asmdata", None)
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


class sve_setup(isa_setup):
    """
    SVE: a governing predicate with every lane active, in the predicate
    register chosen (p0 by default), reserved and aliased as 'all_true' in the
    register tracker
    """

    ALIAS = "all_true"

    def __init__(self, gen, predicate : int = 0):
        super().__init__(gen)
        self.predicate = predicate

    def steps(self) -> list[setup_step]:
        return [setup_step(setup_kind.PREDICATE,
                           f"all-true governing predicate in p{self.predicate} for "
                           f"full-vector instructions")]

    def predication(self) -> predication:
        return predication.EXPLICIT

    def _index(self, rt) -> int:
        aliased = getattr(rt, "aliased_regs", {}).get("mreg", {}) if rt else {}
        return aliased.get(self.ALIAS, self.predicate)

    def setup(self, *, rt, dt, vl : int | None = None) -> str:
        super().setup(rt=rt, dt=dt, vl=vl)
        aliased = getattr(rt, "aliased_regs", {}).get("mreg") if rt else None
        if aliased is not None and self.ALIAS not in aliased:
            rt.reserve_specific_reg("mreg", self.predicate)
            rt.alias_reg("mreg", self.ALIAS, self.predicate)
        return self.gen.set(dreg=self.gen.mreg(self._index(rt)), dt=dt, modifiers={opset_modifier.CONSTMASK},
                            mask="alltrue")

    def all_true_predicate(self, *, rt, dt):
        return self.gen.mreg(self._index(rt))


class sme_setup(sve_setup):
    """
    SME: streaming mode with ZA on and off around the kernel, SVE's predicate,
    and an all-true predicate-as-counter for multi-vector instructions
    """

    COUNTER = 8

    def clobbers(self) -> dict[str, list[int]]:
        """
        smstart and smstop zero the vector registers: every FP register the
        function's caller may expect kept
        """
        return {"freg": list(range(self.gen.max_fregs()))}

    def counter(self) -> bool:
        """
        Whether the predicate-as-counter is set: an SME2 register - with SME2
        enabled, or no features chosen (every one available)
        """
        features = getattr(self.gen, "features", None)
        return features is None or "SME2" in features

    def steps(self) -> list[setup_step]:
        steps = [setup_step(setup_kind.ENGINE, "streaming SVE mode and ZA: smstart, smstop"),
                 *super().steps()]
        if self.counter():
            steps.append(setup_step(setup_kind.PREDICATE,
                                    f"all-true predicate-as-counter in pn{self.COUNTER} for "
                                    f"multi-vector instructions"))
        return steps

    def setup(self, *, rt, dt, vl : int | None = None) -> str:
        out = self.gen.asmwrap("smstart") + super().setup(rt=rt, dt=dt, vl=vl)
        if self.counter():
            out += self.gen.set(dreg=self.gen.mreg(self.COUNTER, True), dt=dt, modifiers={opset_modifier.CONSTMASK},
                                mask="alltrue")
        return out

    def teardown(self, *, rt, dt) -> str:
        return super().teardown(rt=rt, dt=dt) + self.gen.asmwrap("smstop")


class rvv_setup(isa_setup):
    """
    RVV: vsetvli sets the element width, the register grouping (LMUL) and the
    vector length, kept in a register. The vector length can be the most the
    hardware has, or fewer: on some implementations a shorter one performs
    better, and RVV instructions and address increments all use the length set.
    """

    def steps(self) -> list[setup_step]:
        return [setup_step(setup_kind.VECTOR_LENGTH,
                           "vsetvli: element width, grouping and vector length, the "
                           "hardware's or fewer elements")]

    def supports_vl_override(self) -> bool:
        return True

    def predication(self) -> predication:
        return predication.IMPLICIT

    def element_type_is_state(self) -> bool:
        return True

    def setup_dt(self, *, dt, group : int = 1) -> str:
        # LMUL stays the generator's: a register the caller sees is a group
        # of LMUL already, a wide one a group of those
        return self.gen.vsetvli_keep_vl(dt=dt, lmul=getattr(self.gen, "lmul", 1) * group)

    @staticmethod
    def _aliased(rt, name: str) -> int:
        aliased = rt.aliased_regs.setdefault("greg", {})
        if name not in aliased:
            rt.alias_reg("greg", name, rt.reserve_any_reg("greg"))
        return aliased[name]

    def vector_length_register(self, *, rt):
        return self.gen.greg(self._aliased(rt, "vlen"))

    def setup(self, *, rt, dt, vl : int | None = None) -> str:
        super().setup(rt=rt, dt=dt, vl=vl)
        vlreg = self.vector_length_register(rt=rt)
        if vl is not None:
            avl = self.gen.greg(self._aliased(rt, "avl"))
            return self.gen.set(dreg=avl, dt=adt.UINT64, modifiers={opset_modifier.CONST}, value=vl) + \
                self.gen.vsetvli(vlreg=vlreg, avlreg=avl, dt=dt)
        if "avl" in rt.aliased_regs.get("greg", {}):
            # the old interface: the caller has set an aliased avl register
            avl = self.gen.greg(rt.aliased_regs["greg"]["avl"])
            return self.gen.vsetvli(vlreg=vlreg, avlreg=avl, dt=dt)
        return self.gen.vsetvlmax(reg=vlreg, dt=dt)


class constants_setup(isa_setup):
    """
    AVX: constants some instructions read from memory - index data for gathers
    and scatters emulating strided accesses
    """

    def steps(self) -> list[setup_step]:
        return [setup_step(setup_kind.CONSTANTS,
                           "index data for gathers and scatters, written by data() into "
                           "a read-only section", required=False)]
