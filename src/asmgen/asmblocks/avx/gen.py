# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86_64/AVX/FMA asm generator classes

fma128/fma256 use the avx2 implementation, avx512 the avx512 one
"""
from .avx2 import fma128, fma256
from .avx512 import avx512

__all__ = ["fma128", "fma256", "avx512"]
