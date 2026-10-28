# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SME asm generator
"""

from ..op.operand import register_type as rt
from ...registers import (
    asm_data_type as adt,
    reg_tracker,
    adt_size,
    treg_base, vreg_base, greg_base,
)
from ..sve import sve

from .types import sme_treg
from .optransform import sme_fopa
from .opmove import sme_move

from .opmem.load import sme_load
from .opmem.store import sme_store
from .opset import sme_opset
from ..op.opset import opset_modifier as _opset_mod


class sme(sve):
    """
    SME asmgen implementation
    """

    MARCH_BASE = "armv9-a+sme"
    MARCH_EXTENSIONS = sve.MARCH_EXTENSIONS | {"SME2": "sme2", "F64F64": "sme-f64f64", "I16I64": "sme-i16i64",
                                            "F16F16": "sme-f16f16", "SME_B16B16": "sme-b16b16",
                                            "F8F16": "sme-f8f16", "F8F32": "sme-f8f32"}
    # SVE's, and SME's own: FEAT_SME2, FEAT_SME_F64F64, _I16I64, _F16F16,
    # _B16B16, _F8F16, _F8F32
    FEATURES = sve.FEATURES + ("SME2", "F64F64", "I16I64", "F16F16", "SME_B16B16",
                               "F8F16", "F8F32", "FA64")


    def isa_setup(self, predicate : int = 0):
        """
        What this ISA needs set up before a kernel and torn down after

        :param predicate: predicate register for the all-true predicate
        """
        from ..setup import sme_setup
        return sme_setup(self, predicate=predicate)

    def __init__(self):
        super().__init__()
        self.valid_rcregs = [str(self.greg(i)) for i in range(12,16)]
        self.fopa = sme_fopa(asmwrap=self.asmwrap,
                             dt_suffixes=self.dt_suffixes)

        self.load = sme_load(asmwrap = self.asmwrap)
        self.store = sme_store(asmwrap = self.asmwrap)

        self.move = sme_move(asmwrap=self.asmwrap,
                                  dt_suffixes=self.dt_suffixes)
        self.set = sme_opset(asmwrap=self.asmwrap, dt_suffixes=self.dt_suffixes)
        # the operations exist: the features that are off hide their signatures
        self._gate_features()

    def signature_state(self, op, sig) -> tuple[dict[str, str], dict[str, str]]:
        """
        SME: ZA's tiles and slices only in streaming mode with ZA on (sm, za);
        SME2's multi-vector accesses only in streaming mode; gathers and
        scatters only out of it - unless FEAT_SME_FA64, the full A64 set in
        streaming mode (LX2 has it, Apple's parts don't)
        """
        rtypes = {o.rtype for o in sig.operands.values()}
        mods = {m.name for m in sig.modifiers}
        if rt.TILE in rtypes:
            return {"sm": "==1", "za": "==1"}, {}
        if "MULTIVEC" in mods:
            return {"sm": "==1"}, {}
        if "VINDEX" in mods and "FA64" not in (set(self.FEATURES) if self.features is None else set(self.features)):
            return {"sm": "==0"}, {}
        return {}, {}

    def max_tregs(self, dt : adt) -> int:
        return adt_size(dt)
    def max_mregs(self):
        return 16

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        return sme_treg(reg_idx, dt)

    def _c_simd_size_asm(self) -> str:
        # the streaming vector length: counted in streaming mode
        return self.asmwrap("smstart") + super()._c_simd_size_asm() + self.asmwrap("smstop")

