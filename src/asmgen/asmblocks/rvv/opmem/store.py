# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
RVV 1.0 and 0.7.1 store instructions
"""
from collections.abc import Callable

from ...op import opmem_action as action
from .base import rvv_opmem


class rvv_store(rvv_opmem):
    """
    RVV vectore stores
    """

    def __init__(self, asmwrap : Callable[[str],str],
                 lmul_getter :Callable[[],int]):
        super().__init__(action=action.STORE,
                         asmwrap=asmwrap,
                         lmul_getter=lmul_getter)
