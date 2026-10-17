# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Classes for operand restrictions/constraints
"""

from typing import Callable,Type

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .modifier import operation_modifier as mod

from ...registers import (
    data_reg,
    greg_base,
    mreg_base,
    asm_data_type as adt
)

# pylint: disable-next=invalid-name
class ArgumentDependencyError(Exception):
    """
    Validation of an argument can't be performed because it
    depends on another argument that wasn't specified
    """
    def __init__(self, name: str, deps: list[str]):
        self.name = name
        self.deps = deps
        depstr = ", ".join(deps)
        self.msg = f"unset dependency(ies) for {name}: {depstr}"

        super().__init__(self.msg)

# pylint: disable-next=invalid-name
class ConstraintDoesNotApplyError(Exception):
    """
    The constraint does not apply to an argument
    """
    def __init__(self):
        super().__init__("This constraint does not apply in this specific case")

type ValueType = data_reg|greg_base|mreg_base|adt|int

@dataclass
class operand_constraint(ABC):
    """
    Constraint on operand values, like register indices or immediate values
    """
    params: dict[str, ValueType] = field(default_factory=dict, init=False)

    def __post_init__(self):
        pass

    def specialize_params(self,
                          name : str,
                          modifiers : set[mod],
                          context : dict[str,ValueType],
                          params : dict[str,ValueType]):
        """
        modifies context based on specified modifiers and returns it,
        leaving original context unchanged.
        To be overriden by inheriting class.

        :param name: name of the argument for which to specialize params
        :param modifiers: modifiers to apply to the operation
                          (like opd3_modifier.* or opmem_modifier.*)
        :param context: arguments passed to the operation or already
                        assigned arguments when generating valid values
        :param params: constraint-relevant parameters

        :raises ValueError: if the value is invalid or
        :raises ConstraintDoesNotApplyError: The constraint does not apply
            (Example: constraint only applies when a dreg is a vreg)
        """

    def __call__(self, name : str,
                 modifiers : set[mod],
                 val : ValueType,
                 context: dict[str,ValueType]):
        """
        Use this to validate the value for a given operand

        :param name: name of the argument (like 'adreg')
        :param modifiers: modifiers to apply to the operation
                          (like opd3_modifier.* or opmem_modifier.*)
        :param val: value to check for the argument
        :param context: dictionary of already assigned argument values

        :raises ArgumentDependencyError: If the operand depends on the value
            of another operand and it is unset
        :raises ValueError: if the value is invalid
        :raises ConstraintDoesNotApplyError: The constraint does not apply
            (Example: constraint only applies when a dreg is a vreg)
        """

        active_params = self.params.copy()
        if modifiers:
            ref_modifier = next(iter(modifiers))

            if not isinstance(ref_modifier, mod):
                raise ValueError("Modifier type is not operation_modifier")
            if len({type(m) for m in modifiers}) > 1:
                raise ValueError("All modifiers must be of the same type")

            self.specialize_params(
                    name=name,
                    modifiers=modifiers,
                    context=context,
                    params=active_params)

        self.validate(name=name, val=val, context=context, params=active_params)

    @abstractmethod
    def validate(self,
                 name : str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):
        """
        Checks if a value is valid for an argument and raises an Error if it is not.
        To be implemented by an inheriting class

        :param name: name of the argument (like 'adreg')
        :param context: other argument values
        :param params: Constraint parameters

        :raises ArgumentDependencyError: If the operand depends on the value
                                         of another operand and it is unset
        :raises ValueError: if the value is invalid
        """

@dataclass(kw_only=True)
class intval_constraint(operand_constraint):
    """
    Constraint on an integer property of the value

    :param what: human-readable name for the property,
                 used in error messages
    :param getint: Callable retrieving the integer from the value
    :param makeval: Callable constructing a value from an int
    """
    what : str = "value"
    getint : Callable[[ValueType],int] = lambda v : v
    makeval : Callable[[int], ValueType] = lambda d : d

@dataclass(kw_only=True)
class regidx_constraint(intval_constraint):
    """
    Constraint on a register index

    :param reg_class: register type/class, must be constructible
                      with reg_class(reg_idx : int)
    """
    reg_class : Type[greg_base|data_reg|mreg_base]
    what : str = "index"
    getint : Callable[[greg_base|data_reg|mreg_base],int] = lambda reg : reg.idx
    makeval : Callable[[int], greg_base|data_reg|mreg_base] = field(init=False)

    def __post_init__(self):
        self.makeval = lambda idx : self.reg_class(reg_idx=idx)

        super().__post_init__()

@dataclass(kw_only=True)
class minmax_constraint(intval_constraint):
    """
    Constraint limiting an integer value to a range between two
    values inclusively

    :param minval: Smallest allowed value
    :param maxval: Largest allowed value
    """
    minval: int
    maxval: int
    def __post_init__(self):
        self.params['minval'] = self.minval
        self.params['maxval'] = self.maxval

        super().__post_init__()

    def validate(self, name : str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        minval = params['minval']
        maxval = params['maxval']

        if self.getint(val) < minval:
            raise ValueError(f"{self.what} of {name} must be >= {minval}")
        if self.getint(val) > maxval:
            raise ValueError(f"{self.what} of {name} must be <= {maxval}")


@dataclass(kw_only=True)
class oneof_constraint(intval_constraint):
    """
    Constraint limiting an integer value to a known set of values

    :param valset: set of valid integer values
    """
    valset: set[int]
    def __post_init__(self):
        self.params['valset'] = self.valset

        super().__post_init__()

    def validate(self, name : str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        valset = params['valset']

        if self.getint(val) not in valset:
            raise ValueError(f"{self.what} of {name} must be one of: {valset}")


@dataclass(kw_only=True)
class multiple_constraint(intval_constraint):
    """
    Constraint limiting an integer value to multiples of another
    value

    :param multiple: Value the integer has to be a multiple of
    """
    multiple: int
    def __post_init__(self):
        self.params['multiple'] = self.multiple

        super().__post_init__()

    def validate(self, name : str, val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        multiple = params['multiple']

        if 0 != (self.getint(val) % multiple):
            raise ValueError(f"{self.what} of {name} must be a multiple of {multiple}")


@dataclass(kw_only=True)
class otherplusn_constraint(intval_constraint):
    """
    Constraint limiting an integer value to the integer value
    of another argument plus an integer offset

    :param other: name of the dependency
    :param offset: integer offset from the value of the dependency
    """
    other: str
    offset: int
    def __post_init__(self):
        self.params['other'] = self.other
        self.params['offset'] = self.offset

        super().__post_init__()

    def validate(self, name : str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        other = params['other']
        otherint = self.getint(context[other])
        offset = params['offset']

        if self.getint(val) != otherint+offset:
            raise ValueError((f"{self.what} of {name} must be "
                              f"{self.what} of {other} plus {offset}"))


@dataclass(kw_only=True)
class otherplusnmod_constraint(intval_constraint):
    """
    Constraint limiting an integer value to the integer value
    of another argument plus an integer offset modulo another
    integer value

    :param other: name of the dependency
    :param offset: integer offset from the value of the dependency
    :param modval: modulo value to use for wrapping values around
    """
    other: str
    offset: int
    modval: int
    def __post_init__(self):
        self.params['other'] = self.other
        self.params['offset'] = self.offset
        self.params['modval'] = self.modval

        super().__post_init__()

    def validate(self, name : str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        other = params['other']

        if other not in context:
            raise ArgumentDependencyError(name=name, deps=[other])

        otherint = self.getint(context[other])
        offset = params['offset']
        modval = params['modval']

        if self.getint(val) != (otherint+offset) % modval:
            raise ValueError((f"{self.what} of {name} must be "
                              f"{self.what} of {other} plus {offset} "
                              f"modulo {modval}"))

@dataclass(kw_only=True)
class sameval_constraint(intval_constraint):
    """
    Constraint limiting an integer value to the integer value of another argument

    :param other: name of the dependency
    """
    other: str
    def __post_init__(self):
        self.params['other'] = self.other

        super().__post_init__()

    def validate(self, name: str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        other = params['other']

        if other not in context:
            raise ArgumentDependencyError(name=name, deps=[other])

        if self.getint(val) != self.getint(context[other]):
            raise ValueError((f"{self.what} of {name} must be equal to "
                              f"{self.what} of {other}"))

@dataclass(kw_only=True)
class differentval_constraint(intval_constraint):
    """
    Constraint limiting an integer value to be different from the
    integer value of another argument

    :param other: name of the dependency
    """
    other: str
    def __post_init__(self):
        self.params['other'] = self.other

        super().__post_init__()

    def validate(self, name: str,
                 val : ValueType,
                 context : dict[str,ValueType],
                 params : dict[str,ValueType]):

        other = params['other']

        if other not in context:
            raise ArgumentDependencyError(name=name, deps=[other])

        if self.getint(val) == self.getint(context[other]):
            raise ValueError((f"{self.what} of {name} must be different from "
                              f"{self.what} of {other}"))


@dataclass(kw_only=True)
class either_constraint(operand_constraint):
    """
    Constraint satisfied when all constraints of at least one option hold: a
    union of ranges, like AArch64's ldr q immediate - unscaled within
    -256..255 (encoded as ldur) or a multiple of 16 up to 65520
    """
    options : list[list[operand_constraint]]
    what : str = "value"

    def __call__(self, name : str,
                 modifiers : set[mod],
                 val : ValueType,
                 context: dict[str,ValueType]):
        failures = []
        for option in self.options:
            try:
                for constraint in option:
                    try:
                        constraint(name, modifiers, val, context)
                    except ConstraintDoesNotApplyError:
                        pass
                return
            except ValueError as err:
                failures.append(str(err))
        raise ValueError(f"{self.what} of {name} must satisfy one of: " + "; or ".join(failures))

    def validate(self, name : str, val : ValueType,
                 context : dict[str,ValueType], params : dict[str,ValueType]):
        pass
