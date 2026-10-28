# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# ------------------------------------------------------------------------------
"""
AVX's setup: constants some instructions read from memory
"""
from __future__ import annotations

from ..setup import isa_setup, setup_kind, setup_step


class constants_setup(isa_setup):
    """
    AVX: constants some instructions read from memory - index data for gathers
    and scatters emulating strided accesses
    """

    def steps(self) -> list[setup_step]:
        return [setup_step(setup_kind.CONSTANTS,
                           "index data for gathers and scatters, written by data() into "
                           "a read-only section", required=False)]
