# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V RVV 0.7.1 asm generator and related types
"""

from .setup import rvv071_setup
from ..noarch import compiler_family
from ..rvv import rvv

from ...registers import asm_data_type as adt, adt_size, greg_base, vreg_base
from ...registers import asm_index_type as ait
from ..rvv.opset import rvv_opset
from .opmem import rvv071_load, rvv071_store
from .opmove import rvv071_move

def _th_prefixed(line : str) -> str:
    """
    A vector instruction as XTheadVector spells it - th.vle.v for vle.v -
    other lines (scalar instructions, labels) as they are
    """
    stripped = line.lstrip()
    token = stripped.split()[0] if stripped.split() else ""
    if token.startswith("v") and not token.endswith(":"):
        return line[:len(line) - len(stripped)] + "th." + stripped
    return line


# pylint: disable=too-many-public-methods
class rvv071(rvv):
    """
    RISC-V RVV 0.7.1 asmgen implementation
    """

    #: XTheadVector: 0.7.1's vector instructions th.-prefixed, as mainline
    #: binutils and GCC take them; off, unprefixed (EPI's toolchain, -mepi)
    ENCODING_FEATURES = {"XTheadVector": True}

    def compiler_flags(self, compiler : str = "gcc") -> list[str]:
        """
        XTheadVector's -march (GCC, LLVM); without it EPI's toolchain's
        -mepi, which takes the unprefixed forms
        """
        if not self._encoding_features()["XTheadVector"]:
            return ["-mepi"]
        if "arm" == compiler_family(compiler):
            raise NotImplementedError("armclang targets no RISC-V")
        extensions = ["zba"] if "ZBA" in self.enabled_features() else []
        return ["-march=" + "_".join(["rv64gc"] + extensions + ["xtheadvector"])]

    def isa_setup(self):
        """
        What this ISA needs set up before a kernel and torn down after: 0.7.1's
        vsetvli
        """
        return rvv071_setup(self)

    def asmwrap(self, code : str) -> str:
        """
        The instructions wrapped as rvv's are, their vector mnemonics
        th.-prefixed under XTheadVector
        """
        if self._encoding_features()["XTheadVector"]:
            code = "\n".join(_th_prefixed(line) for line in code.split("\n"))
        return super().asmwrap(code)

    def __init__(self):
        super().__init__()
        # vsetvli without tail and mask policies
        self.set = rvv_opset(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul, policy=False)
        # 0.7.1's mnemonics (0.6, phase 5)
        self.load = rvv071_load(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul)
        self.store = rvv071_store(asmwrap=self.asmwrap, lmul_getter=lambda : self.lmul)
        self.move = rvv071_move(asmwrap=self.asmwrap)
        # the operations exist: the features that are off hide their signatures
        self._gate_features()

    #: the mask population count
    vpopc = "vmpopc.m"

    dt_suffixes = {
            adt.DOUBLE  : "e",
            adt.SINGLE  : "w",
            adt.HALF    : "h",
            adt.FP8E4M3 : "b",
            adt.FP8E5M2 : "b",
            }
