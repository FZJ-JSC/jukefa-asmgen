# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISCV64 +D/F prefetch instructions (Zicbop)
"""

from collections.abc import Callable

from ...op import operand_modifier as opd_mod
from ...op import opmem_action as action
from ...op import opmem_modifier as mod
from ..types import riscv64_greg
from .base import riscv64_opmem
from .signatures import make_riscv64_prefetch_signatures


class riscv64_prefetch(riscv64_opmem):
    """
    RISCV64 +D/F prefetch (prefetch.r).

    Data-less opmem (no dregs/dt). Prefetch target (read/write) and
    cache level/locality selection are not exposed yet.
    """

    def __init__(self,
                 asmwrap : Callable[[str],str]):
        super().__init__(action=action.PREFETCH, asmwrap=asmwrap)
        self.signatures = make_riscv64_prefetch_signatures()

    # pylint: disable-next=arguments-differ
    def implementation(self, *, agreg: riscv64_greg,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:
        del operand_modifiers

        addressing = self.get_addressing(agreg, modifiers, **kwargs)
        return self.asmwrap(f"prefetch.r {addressing}")