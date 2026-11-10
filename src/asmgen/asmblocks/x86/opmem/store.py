# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base X86 store instructions
"""

from collections.abc import Callable

from ...op import opmem_action as action
from .base import x86_opmem


class x86_store(x86_opmem):
    """
    Base X86 freg and greg stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)
