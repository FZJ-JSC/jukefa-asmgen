# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
C++ expressions
"""
from enum import Enum, unique


def identity(c_var):
    """
    Expression that evaluates to the variable that was passed in
    """
    return f"({c_var})"

def add(c_var1, c_var2):
    """
    Expression that evaluates to the sum of 2 variables
    """
    return f"({c_var1}+{c_var2})"

def mul(c_var1, c_var2):
    """
    Expression that evaluates to the product of 2 variables
    """
    return f"({c_var1}*{c_var2})"

def div(c_var1, c_var2):
    """
    Expression that evaluates to the division of 2 variables
    """
    return f"({c_var1}/{c_var2})"

@unique
class shift_direction(Enum):
    """
    Bit shift direction (LEFT/RIGHT)
    """
    LEFT=0
    RIGHT=1

class shiftlit:
    """
    Expression that evaluates to the bit-shifted value of a variable
    The bit-shift-count comes from an immediate/literal
    """
    def __init__(self, direction, bit_count=1):
        self.shiftop = "<<" if (direction == shift_direction.LEFT) else ">>"
        self.bit_count = bit_count
    def __call__(self,c_var):
        return f"({c_var} {self.shiftop} {self.bit_count})"

class shiftvar:
    """
    Expression that evaluates to the bit-shifted value of a variable
    The bit-shift-count comes from another variable
    """
    def __init__(self, direction):
        self.shiftop = "<<" if (direction == shift_direction.LEFT) else ">>"
    def __call__(self, c_var1, c_var2):
        return f"({c_var1} {self.shiftop} {c_var2})"
