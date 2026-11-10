# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE load instructions
"""

from collections.abc import Callable

from ...op import opmem_action as action
from .base import sve_opmem


class sve_load(sve_opmem):
    """
    SVE register loads
    """

    bcast_supported=True

    def __init__(self, asmwrap : Callable[[str],str]):
        super().__init__(action=action.LOAD, asmwrap=asmwrap)
