# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base AArch64 prefetch instructions
"""

from collections.abc import Callable

from ...op import operand_modifier as opd_mod
from ...op import opmem_action as action
from ...op import opmem_modifier as mod
from ..types import aarch64_greg
from .base import aarch64_opmem
from .signatures import make_aarch64_prefetch_signatures


class aarch64_prefetch(aarch64_opmem):
    """
    Base AArch64 prefetch (prfm PLDL1KEEP).

    Data-less opmem (no dregs/dt). Cache level/locality/target selection
    is not exposed yet.
    """

    def __init__(self,
                 asmwrap : Callable[[str],str]):
        super().__init__(action=action.PREFETCH, asmwrap=asmwrap)
        self.signatures = make_aarch64_prefetch_signatures()

    # pylint: disable-next=arguments-differ
    def implementation(self, *, agreg: aarch64_greg,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:
        del operand_modifiers

        addressing = self.get_addressing(agreg, modifiers, **kwargs)
        return self.asmwrap(f"prfm PLDL1KEEP, {addressing}")