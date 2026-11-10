# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base class for ASM test generators
"""
from .generators.greg import greg_test_generator
from .generators.vec import vec_test_generator


class asm_test_generator(vec_test_generator, greg_test_generator):
    """
    Base class for ASM test generators, providing functionality by
    inheriting from specific test generators
    """
