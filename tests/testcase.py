# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Type hinting helper for asmgen test cases
"""

from asmgen.asmblocks.noarch import asmgen


# This exists just for type hinting
class testcase:
    """
    Type hinting helper for asmgen test cases
    """
    gen : asmgen
    cxx_name : str
    name : str
    allowed : list[str]
