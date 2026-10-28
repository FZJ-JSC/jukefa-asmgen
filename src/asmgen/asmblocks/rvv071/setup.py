# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
RVV 0.7.1's setup: RVV's, its vsetvli spelt as 0.7.1 has it
"""
from __future__ import annotations

from ..rvv.setup import rvv_setup


class rvv071_setup(rvv_setup):
    """
    RVV 0.7.1 (XTheadVector): vsetvli as 1.0's, without the tail and mask
    policies, which 0.7.1 doesn't have; th.-prefixed where the generator's
    asmwrap does it
    """
    POLICIES = ""
