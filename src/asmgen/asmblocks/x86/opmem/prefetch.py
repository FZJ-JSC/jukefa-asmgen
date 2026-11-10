# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base X86 prefetch instructions
"""

from collections.abc import Callable

from ...op import operand_modifier as opd_mod
from ...op import opmem_action as action
from ...op import opmem_modifier as mod
from ..types import x86_greg
from .base import x86_opmem
from .signatures import make_x86_prefetch_signatures


class x86_prefetch(x86_opmem):
    """
    Base X86 prefetch (prefetcht0).

    Data-less opmem (no dregs/dt). Cache level/locality selection is not
    exposed yet.
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.PREFETCH, asmwrap=asmwrap, rpref=rpref)
        self.signatures = make_x86_prefetch_signatures()

    # pylint: disable-next=arguments-differ
    def implementation(self, *, agreg: x86_greg,
                       modifiers: set[mod],
                       operand_modifiers : dict[str,set[opd_mod]],
                       **kwargs) -> str:
        del operand_modifiers

        addressing = self.get_addressing(agreg, modifiers, **kwargs)
        return self.asmwrap(f"prefetcht0 {addressing}")