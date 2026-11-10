# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Classes to store various constant data to be used with ASM instructions
"""

import struct
from dataclasses import dataclass

from .registers import adt_is_float, adt_size
from .registers import asm_data_type as adt


def adt_to_type_macro(dt : adt) -> str:
    """
    Converts an abstract data type (adt) to its corresponding assembly 
    size macro representation.

    This helps to determine the appropriate assembly directive for 
    data declaration based on the size of the specified data type.

    :param dt: The abstract data type for which to get the macro
    :type dt: class:`asmgen.registers.asm_data_type`
    :return: The corresponding assembly directive as a string
    :rtype: str
    :raises ValueError: If there is no assembly data macro corresponding to the given adt
    """

    size = adt_size(dt)

    if size == 1:
        return ".byte"
    if size == 2:
        return ".short"
    if size == 4:
        return ".long"
    if size == 8:
        return ".quad"

    raise ValueError(f"No ASM data macro for adt {dt}")

def adt_fp_bits(dt: adt) -> tuple[int,int,bool]:
    """
    Return exponent bits, mantissa bits, and sign bit for given floating-point data type.

    :param dt: data type
    :type dt: class:`asmgen.registers.asm_data_type`
    """

    ems_map = {
        adt.FP8E4M3: (4,3,True),
        adt.FP8E5M2: (5,2,True),
        adt.BF16:    (8,7,True),
        adt.FP16:    (5,10,True),
        adt.FP32:    (8,23,True),
        adt.TF32:    (8,23,True), # Same as FP32, reduced precision is only for compute
        adt.FP64:    (11,52,True),
        adt.FP128:   (15,113,True),
    }

    if dt not in ems_map:
        raise ValueError(f"{dt} is not a floating-point type.")

    return ems_map[dt]

def get_fp_hex_value(value : float, ebits: int, mbits: int, signbit : bool = True) -> str:
    """
    Converts a floating-point value into its hexadecimal representation,
    extracting the specified number of exponent bits and mantissa bits,
    along with an optional sign bit.

    :param value: The floating-point value to convert
    :type value: float
    :param ebits: Number of exponent bits
    :type ebits: int
    :param mbits: Number of mantissa bits
    :type mbits: int
    :param signbit: Indicates whether to include a sign bit (default is True)
    :type signbit: bool
    :return: The hexadecimal representation of the modified floating-point value
    :rtype: str
    :raises ValueError: If the total number of bits (sign + exponent + mantissa) 
                        is not 8, 16, 32, or 64
    """

    # should fit into a byte,short,long or quad
    if ebits+mbits+(1 if signbit else 0) not in [8,16,32,64]:
        raise ValueError("Total number of bits has to be either 8,16,32 or 64")


    result_int = 0

    bits = struct.unpack('>Q', struct.pack('>d', value))[0]


    # mantissa
    result_int |= (bits & ((1 << mbits) - 1))
    bits >>= mbits


    # exponent
    result_int |= ((bits & ((1 << ebits) -1)) << mbits)
    bits >>= ebits

    # sign
    if signbit:
        result_int |= ((bits & 0x1) << mbits+ebits)

    return hex(result_int)

@dataclass
class asm_data:
    """
    Represents a data type for assembly data, which can store either an 
    integer or a floating-point value along with its associated type.

    :param dt: The data type of the value (e.g. INT8, FP32, ...)
    :type dt: class:`asmgen.registers.asm_data_type`
    :param value: The actual value of the data, which can be an int or float
    :type value: Union[int, float]
    """
    dt : adt
    value : int|float


    def __str__(self) -> str:
        """Converts the asm_data instance to its corresponding assembly representation."""
        tmacro = adt_to_type_macro(self.dt)

        vhex : str = ""
        if adt_is_float(self.dt):
            ebits,mbits,signbit = adt_fp_bits(self.dt)
            vhex = str(get_fp_hex_value(self.value, ebits, mbits, signbit))
        else:
            bits = adt_size(self.dt)*8
            value = int(self.value) & ((1 << bits) -1 )
            vhex = str(hex(value))

        return f"{tmacro} {vhex}"
