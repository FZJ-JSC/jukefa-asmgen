# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX/FMA/AVX2/AVX512 load instructions
"""

from collections.abc import Callable

from ...op import opmem_action as action
from .base import avx128_opmem, avx256_opmem, avx512_opmem


class avx128_load(avx128_opmem):
    """
    AVX2 128bit loads
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.LOAD, asmwrap=asmwrap, rpref=rpref)

class avx256_load(avx256_opmem):
    """
    AVX2 256bit loads
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.LOAD, asmwrap=asmwrap, rpref=rpref)

class avx512_load(avx512_opmem):
    """
    AVX512 loads
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.LOAD, asmwrap=asmwrap, rpref=rpref)
