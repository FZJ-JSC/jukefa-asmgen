# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE opd3 base case
"""

from typing import Callable

from ....registers import (
    asm_data_type as adt,
)

from ...aarch64.opd3.base import aarch64_simd_opd3_base

from .signatures import make_sve_opd3_signatures

class sve_opd3_base(aarch64_simd_opd3_base):
    """
    SVE base opd3 implementation with methods shared by all
    opd3 operations
    """

    inst_base = "invalid"

    supports_np = False
    # predicated forms overwrite their first source: Zdn
    destructive = False
    # unpredicated forms (vectors) exist besides the predicated ones
    unpredicated = False
    # floating-point widening exists as FMA only
    widens = True

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 dt_idxsuffixes : dict[adt,str]
                 ):
        super().__init__(asmwrap,dt_suffixes,dt_idxsuffixes)

        self.signatures = make_sve_opd3_signatures(self.supports_np, self.destructive,
                                                   self.unpredicated, self.widens)
