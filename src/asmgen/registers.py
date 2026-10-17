# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Contains facilities to work with different ASM register types independently from the ISA
"""
from abc import ABC,abstractmethod
from enum import Enum,auto
from typing import Union

from .util import NIE_MESSAGE

class reg_tracker:
    """
    Allocates registers by specified type and tracks their state,
    as well as if they were clobbered.
    Able to alias registers with a string alias

    :param registered_types : register types known to the tracker
    :type registered_types : set[str]
    :param max_regs : maximum available registers for each type
    :type max_regs : dict[str,int]
    :param available_regs : list of existing registers for each type
    :type available_regs : dict[str,list[int]]
    :param used_regs : registers currently in use for each type
    :type used_regs : dict[str,set[int]]
    :param clobbered_regs : registers that have been in use for each type
    :type clobbered_regs : dict[str,set[int]]
    """

    # is accessed read-only, should be fine
    # pylint: disable=dangerous-default-value
    def __init__(self, reg_type_init_list : list[tuple[str,int]] = []):
        """
        Constructor method

        :param reg_type_init_list : list of tuples containing ("type_tag", max_regs)
            that the initial attributes will be initialized from
        :type reg_type_init_list : list[tuple[str,int]], optional
        """

        self.registered_types : set[str] = set()

        self.max_regs       : dict[str,int] = {}
        self.available_regs : dict[str,list[int]] = {}
        self.used_regs      : dict[str,set[int]] = {}
        self.clobbered_regs : dict[str,set[int]] = {}
        self.aliased_regs   : dict[str,dict[str,int]] = {}
        # type -> [(other type, index in this type -> index in the other)]
        self.shared         : dict[str,list[tuple[str,dict[int,int]]]] = {}

        for tag,max_regs in reg_type_init_list:
            self.add_type(type_tag=tag, max_regs=max_regs)

    def add_type(self, type_tag : str, max_regs : int):
        """
        Adds a single new register type to track

        :param type_tag : tag/type name of the register to track
        :type type_tag : str
        :param max_regs : maximum number of available registers of this type
        :type max_regs : int
        """
        if type_tag in self.registered_types:
            raise ValueError(f"Type already tracked: {type_tag}")
        self.registered_types.add(type_tag)
        self.max_regs[type_tag] = max_regs
        self.available_regs[type_tag] = list(range(max_regs))
        self.clobbered_regs[type_tag] = set()
        self.used_regs[type_tag] = set()
        self.aliased_regs[type_tag] = {}

    def reset(self):
        """
        Resets the tracked state of all registers, clearing used and clobbered registers
        """
        for tag in self.registered_types:
            self.clobbered_regs[tag] = set()
            self.used_regs[tag] = set()

    def share(self, type_a : str, type_b : str, mapping : dict[int,int] | None = None):
        """
        Declare registers of two types to be the same registers: AArch64's d0
        is the low part of v0, RVV's mask register is vector register 0.
        Reserving or releasing one then reserves or releases the other.

        :param mapping: index in type_a -> index in type_b; None for the same
                        index, for every register both types have
        """
        for tag in (type_a, type_b):
            if tag not in self.registered_types:
                raise ValueError(f"Type not tracked: {tag}")
        if mapping is None:
            mapping = {i: i for i in range(min(self.max_regs[type_a], self.max_regs[type_b]))}
        self.shared.setdefault(type_a, []).append((type_b, dict(mapping)))
        self.shared.setdefault(type_b, []).append((type_a, {v: k for k, v in mapping.items()}))

    def _counterparts(self, type_tag : str, idx : int) -> list[tuple[str,int]]:
        return [(other, m[idx]) for other, m in self.shared.get(type_tag, []) if idx in m]

    def _free(self, type_tag : str, idx : int) -> bool:
        return idx not in self.used_regs[type_tag] and \
            all(i not in self.used_regs[t] for t, i in self._counterparts(type_tag, idx))

    def _mark(self, type_tag : str, idx : int):
        for tag, i in [(type_tag, idx)] + self._counterparts(type_tag, idx):
            self.used_regs[tag].add(i)
            self.clobbered_regs[tag].add(i)

    def reserve_any_reg(self, type_tag : str) -> int:
        """
        Reserves a single register of the specified type and marks it as used and clobbered

        :param type_tag : tag/type name of the register
        :type type_tag : str
        :return : index of the reserved register
        :rtype : int
        """
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        # registers no other type shares first: RVV's v0, also the mask
        # register, comes last; with every register shared the order stays
        ordered = sorted(self.available_regs[type_tag],
                         key=lambda i: bool(self._counterparts(type_tag, i)))
        for idx in ordered:
            if self._free(type_tag, idx):
                self._mark(type_tag, idx)
                return idx
        raise IndexError(f"All {type_tag} registers in use!")

    def alias_reg(self, type_tag : str, name : str, idx : int):
        """
        Creates a string alias for the specified register that can later be used to
        retrieve the index

        :param type_tag : tag/type name of the register
        :type type_tag : str
        :param name : Alias name to give to this register
        :type name : str
        :param idx : register index
        :type idx : int
        """
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        if not idx in self.used_regs[type_tag]:
            raise IndexError(f"can't alias unused {type_tag} register nr. {idx}")
        # Throw error if an alias already exists.
        # Theoretically I don't see an issue using multiple aliases
        # for the same register, but I feel just allowing it will lead
        # to some nasty bugs
        if idx in self.aliased_regs[type_tag].values():
            alias_index = list(self.aliased_regs[type_tag].values()).index(idx)
            alias = list(self.aliased_regs[type_tag].keys())[alias_index]
            raise IndexError(f"can't alias {type_tag} register nr. {idx}, "
                             f"it already has the alias \"{alias}\"")
        self.aliased_regs[type_tag][name] = idx

    def reserve_specific_reg(self, type_tag : str, idx : int):
        """
        Reserves the specified register, marking it as used and clobbered

        :param type_tag : tag/type name of the register
        :type type_tag : str
        :param idx : index of the register to reserve
        :type idx : int
        """
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        if not self._free(type_tag, idx):
            raise IndexError(f"{type_tag} register nr. {idx} already in use")
        if idx >= self.max_regs[type_tag]:
            raise IndexError((f"{type_tag} register nr. {idx} not accessible"
                              f"(max_regs = {self.max_regs[type_tag]})"))

        self._mark(type_tag, idx)

    def unuse_reg(self, type_tag : str, idx : int):
        """
        Unmarks the specified register, allowing it to be reserved again

        :param type_tag : tag/type name of the register
        :type type_tag : str
        :param idx : index of the register to unmark
        :type idx : int
        """
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        # Check if an alias exists and remove it.
        # Purpose is to get an error if an alias is used
        # after the register was freed up
        if idx in self.aliased_regs[type_tag].values():
            alias_index = list(self.aliased_regs[type_tag].values()).index(idx)
            alias = list(self.aliased_regs[type_tag].keys())[alias_index]
            del self.aliased_regs[type_tag][alias]
        self.used_regs[type_tag].remove(idx)
        # the same register under its other types is free again too
        for tag, i in self._counterparts(type_tag, idx):
            self.used_regs[tag].discard(i)

    def available_reg_count(self, type_tag : str) -> int:
        """
        Returns the number of available registers of the specified type
        """
        return len(self.available_regs[type_tag])

    def used_reg_count(self, type_tag : str) -> int:
        """
        Returns the number of registers of the specified type that are currently in use
        """
        return len(self.used_regs[type_tag])

    def get_used_regs(self, type_tag : str) -> set[int]:
        """
        Returns the list of register indices of the specified type that are currently in use
        """
        return self.used_regs[type_tag]

    def get_clobbered_regs(self, type_tag : str) -> set[int]:
        """
        Returns the list of register indices of the specified type that were clobbered
        """
        return self.clobbered_regs[type_tag]


class asm_data_type(Enum):
    """
    Data type for assembly instruction specialization and size/address computations
    """
    FP8E4M3 = auto()
    FP8E5M2 = auto()
    UINT8   = auto()
    SINT8   = auto()
    BF16    = auto()
    HALF    = auto()
    FP16    = HALF
    UINT16  = auto()
    SINT16  = auto()
    TF32    = auto()
    XF32    = TF32
    SINGLE  = auto()
    FP32    = SINGLE
    UINT32  = auto()
    SINT32  = auto()
    DOUBLE  = auto()
    FP64    = DOUBLE
    UINT64  = auto()
    SINT64  = auto()
    FP128   = auto()

def adt_size(dt : asm_data_type) -> int:
    """
    Returns the storage size of a data type

    :param dt : data type to return the storage size for
    :type dt : class:`asmgen.registers.asm_data_type`
    :return : Storage size in bytes
    :rtype : int
    """
    size_map = {
        asm_data_type.FP8E4M3 : 1,
        asm_data_type.FP8E5M2 : 1,
        asm_data_type.UINT8   : 1,
        asm_data_type.SINT8   : 1,
        asm_data_type.BF16    : 2,
        asm_data_type.HALF    : 2,
        asm_data_type.UINT16  : 2,
        asm_data_type.SINT16  : 2,
        asm_data_type.TF32    : 4,
        asm_data_type.SINGLE  : 4,
        asm_data_type.UINT32  : 4,
        asm_data_type.SINT32  : 4,
        asm_data_type.DOUBLE  : 8,
        asm_data_type.UINT64  : 8,
        asm_data_type.SINT64  : 8,
        asm_data_type.FP128   : 16,
    }
    if dt not in size_map:
        raise RuntimeError(f"Invalid asm_data_type: {dt}")
    return size_map[dt]

def adt_is_float(dt : asm_data_type) -> bool:
    """
    Checks if the data type is a floating point data type

    :param dt : data type to return the storage size for
    :type dt : class:`asmgen.registers.asm_data_type`
    :return : True if dt is a floating point type, False otherwise
    :rtype : bool
    """
    adt = asm_data_type
    fp_types = [adt.FP128, adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
    return dt in fp_types

def adt_is_int(dt : asm_data_type) -> bool:
    """
    Checks if the data type is an integer data type

    :param dt : data type to return the storage size for
    :type dt : class:`asmgen.registers.asm_data_type`
    :return : True if dt is an integer type, False otherwise
    :rtype : bool
    """
    adt = asm_data_type
    int_types = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8,
                 adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
    return dt in int_types

def adt_is_signed(dt : asm_data_type) -> bool:
    """
    Checks if the data type is a signed integer data type

    :param dt : data type to return the storage size for
    :type dt : class:`asmgen.registers.asm_data_type`
    :return : True if dt is a signed integer type, False otherwise
    :rtype : bool
    """
    adt = asm_data_type
    int_types = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
    return dt in int_types

def adt_is_unsigned(dt : asm_data_type) -> bool:
    """
    Checks if the data type is an unsigned integer data type

    :param dt : data type to return the storage size for
    :type dt : class:`asmgen.registers.asm_data_type`
    :return : True if dt is an unsigned integer type, False otherwise
    :rtype : bool
    """
    adt = asm_data_type
    int_types = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]
    return dt in int_types

class adt_triple:
    """
    Class containing data types for 3 components. This is used for arithmetic 
    instructions with 3 operands like FMA or DOTA instructions as well as for
    calculating data-type specific offsets, etc...

    :param a : data type of component a
    :type a : class:`asmgen.registers.asm_data_type`
    :param b : data type of component b
    :type b : class:`asmgen.registers.asm_data_type`
    :param c : data type of component c
    :type c : class:`asmgen.registers.asm_data_type`
    """
    def __init__(self,
                 a_dt : asm_data_type,
                 b_dt : asm_data_type,
                 c_dt : asm_data_type):
        """
        Constructor method

        :param a_dt : data type of component a
        :type a_dt : class:`asmgen.registers.asm_data_type`
        :param b_dt : data type of component b
        :type b_dt : class:`asmgen.registers.asm_data_type`
        :param c_dt : data type of component c
        :type c_dt : class:`asmgen.registers.asm_data_type`
        """
        self.a = a_dt
        self.b = b_dt
        self.c = c_dt

    def __getitem__(self, key : Union[int,str]):
        if key in [0,'a']:
            return self.a
        if key in [1,'b']:
            return self.b
        if key in [2,'c']:
            return self.c

        raise ValueError(f"invalid triple key: {key}")

    def __eq__(self, other : object) -> bool:
        if not isinstance(other, adt_triple):
            return NotImplemented
        return (self.a == other.a) and\
               (self.b == other.b) and\
               (self.c == other.c)
    def __repr__(self) -> str:
        return f"{str(self.a), str(self.b), str(self.c)}"

class asm_index_type(Enum):
    """
    Index type for assembly instruction specialization and size/address computations
    """
    INT8    = auto()
    INT16   = auto()
    INT32   = auto()
    INT64   = auto()

def ait_size(dt : asm_index_type) -> int:
    """
    Returns the storage size of an index type

    :param dt : index type to return the storage size for
    :type dt : class:`asmgen.registers.asm_index_type`
    :return : Storage size in bytes
    :rtype : int
    """
    size_map = {
        asm_index_type.INT8   : 1,
        asm_index_type.INT16  : 2,
        asm_index_type.INT32  : 4,
        asm_index_type.INT64  : 8,
    }
    if dt not in size_map:
        raise RuntimeError(f"Invalid asm_index_type: {dt}")
    return size_map[dt]

def it_from_dt_samesize(dt : asm_data_type) -> asm_index_type:
    """
    Returns an index type with the same size as the data type

    :param dt : data type to return the index type for
    :type dt : class:`asmgen.registers.asm_data_type`
    :return : Index type
    :rtype : class:`asmgen.registers.asm_index_type`
    :raises: ValueError if dt has no corresponding index type
    """
    if adt_size(dt) == 1:
        return asm_index_type.INT8
    if adt_size(dt) == 2:
        return asm_index_type.INT16
    if adt_size(dt) == 4:
        return asm_index_type.INT32
    if adt_size(dt) == 8:
        return asm_index_type.INT64

    raise ValueError(f"dt {dt} has no corresponding it")

# pylint: disable=too-few-public-methods

class greg_base(ABC):
    """
    Base class for general purpose registers
    """
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError(NIE_MESSAGE)
    @abstractmethod
    def __str__(self):
        """
        Return the string representation of the register
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def idx(self):
        """
        Return the numeric index of the register
        """


class mreg_base(ABC):
    """
    Base class for mask registers
    """
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError(NIE_MESSAGE)
    @abstractmethod
    def __str__(self):
        """
        Return the string representation of the register
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def idx(self):
        """
        Return the numeric index of the register
        """

class data_reg(ABC):
    """
    Base class for data registers (vector/scalar/tile/...)
    """
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError(NIE_MESSAGE)
    @abstractmethod
    def __str__(self):
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def idx(self):
        """
        Return the numeric index of the register
        """

class freg_base(data_reg):
    """
    Base class for scalar registers
    """

class vreg_base(data_reg):
    """
    Base class for vector registers
    """

class treg_base(data_reg):
    """
    Base class for tile registers
    """

# pylint: enable=too-few-public-methods
