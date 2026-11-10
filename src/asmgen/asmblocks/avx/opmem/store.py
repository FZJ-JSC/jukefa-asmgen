# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX/FMA/AVX2/AVX512 store instructions
"""

from collections.abc import Callable
from typing import Any

from ....registers import asm_data_type as adt
from ...op import operand_modifier as opd_mod
from ...op import opmem_action as action
from ...op.opmem import opmem_modifier as mod
from .base import avx128_opmem, avx256_opmem, avx512_opmem, avx_opmem


class no_bcast(avx_opmem):
    """
    Helper class for diagnosing lack of BCAST support
    """
    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs,dts)

        if any(opd_mod.BCAST in mods for _,mods in operand_modifiers.items()):
            raise ValueError("BCAST modifier can't be used with stores")

class no_vindex(avx_opmem):
    """
    Helper class for diagnosing lack of VINDEX support
    """
    def diagnose_failure(self, modifiers : set[mod],
                         operand_modifiers : dict[str,set[opd_mod]],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]):
        super().diagnose_failure(modifiers, operand_modifiers, kwargs,dts)

        if mod.VINDEX in modifiers:
            raise ValueError("VINDEX modifier can't be used with avx2 128/256 bit stores")

class avx128_store(avx128_opmem,no_bcast,no_vindex):
    """
    AVX2 128bit stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)


# it's fine
# pylint: disable-next=too-many-ancestors
class avx256_store(avx256_opmem,no_bcast,no_vindex):
    """
    AVX2 256bit stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)

class avx512_store(avx512_opmem,no_bcast):
    """
    AVX512 stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)
