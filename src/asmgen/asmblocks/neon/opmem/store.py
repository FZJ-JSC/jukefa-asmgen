# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON store instructions
"""
from collections.abc import Callable

from ...op import opmem_action as action
from .base import neon_opmem


class neon_store(neon_opmem):
    """
    NEON register stores
    """

    def __init__(self, asmwrap : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap)
