# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Abstract base classes for arithmetic operations/instructions
"""

from abc import ABC,abstractmethod

from ...registers import (
    data_reg,
    greg_base,
    mreg_base,
    asm_data_type as adt,
)

from .signature import operation_signature
from .misc import make_ord_prefix
from .modifier import operation_modifier as mod
from .operand import operand_modifier as opd_mod

class operation(ABC):
    """
    Abstraction over operations/instructions
    """

    NIE_MESSAGE="Inheriting class must implement this method"

    #: optional ISA extensions enabled - signatures requiring others are
    #: hidden; None: no gating, every signature
    enabled_features : frozenset | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        raw = cls.__dict__.get("get_signatures")
        if raw is not None and not getattr(raw, "feature_gated", False):
            def get_signatures(self, *args, _raw=raw, **kw):
                sigs = _raw(self, *args, **kw)
                enabled = self.enabled_features
                if enabled is None:
                    return sigs
                return [s for s in sigs if getattr(s, "requires", frozenset()) <= enabled]
            get_signatures.feature_gated = True
            get_signatures.__doc__ = raw.__doc__
            cls.get_signatures = get_signatures

    @abstractmethod
    def get_signatures(self) -> list[operation_signature]:
        """
        Returns a list of operation signatures this operation supports

        :return: List of valid signatures
        """

    def diagnose_failure(self,
                         modifiers: set[mod],
                         operand_modifiers: dict[str,set[opd_mod]],
                         kwargs: dict, dts: dict[str, adt]):
        """
        Optional hook for inheriting classes to provide domain-specific 
        error messages.

        :raises ValueError: if a specific bad pattern is found.
        """

    def _auto_diagnose(self,
                       modifiers: set[mod],
                       operand_modifiers: dict[str,set[opd_mod]],
                       kwargs: dict, dts: dict[str, adt]):
        """
        Automatically infers basic errors by looking at the pooled signatures.
        """

        # Not using these for auto diagnosis right now
        del kwargs, dts

        sigs = self.get_signatures()

        all_supported_mods = set().union(*(sig.modifiers for sig in sigs))
        unsupported_mods = modifiers - all_supported_mods
        if unsupported_mods:
            unsup_mod_string = "|".join(m.name for m in unsupported_mods)
            raise ValueError(
                    (f"{type(self).__name__} does not support "
                     f"these modifiers at all: {{{unsup_mod_string}}}"))

        all_supported_operand_mods = set().union(
                *(op.modifiers for sig in sigs for _,op in sig.operands.items()))
        all_operand_mods = set().union(*(mods for _,mods in operand_modifiers.items()))

        unsupported_mods = all_operand_mods - all_supported_operand_mods
        if unsupported_mods:
            print(f"MODS: {all_operand_mods}")
            print(f"SUP MODS: {all_supported_operand_mods}")
            unsup_mod_string = "|".join(m.name for m in unsupported_mods)
            raise ValueError(
                    (f"{type(self).__name__} does not support "
                     f"these modifiers at all: {{{unsup_mod_string}}}"))


    def resolve_dregs(self,
                      dregs : list[data_reg],
                      dts : dict[str,adt]) -> dict[str,data_reg]:
        """
        Helper to create named dreg arguments

        :param dregs: list of dregs
        :return: dictionary mapping argument names to registers
        """

        resolved_operands = {}

        for i,reg in enumerate(dregs):
            pfx = make_ord_prefix(i)
            dreg_name = f"{pfx}dreg"
            dt_name = f"{pfx}_dt"
            resolved_operands[dreg_name] = reg
            resolved_operands[dt_name] = dts.get(dreg_name)

        return resolved_operands

    def resolve_gregs(self, gregs : list[greg_base]) -> dict[str,greg_base]:
        """
        Helper to create named greg arguments

        :param dregs: list of dregs
        :return: dictionary mapping argument names to registers
        """

        resolved_operands = {}

        for i,reg in enumerate(gregs):
            pfx = make_ord_prefix(i)
            greg_name = f"{pfx}greg"
            resolved_operands[greg_name] = reg

        return resolved_operands


    # Still ok, but consider pulling out the diagnosis stuff into a separate
    # function if anything else gets added
    # pylint: disable-next=too-many-locals
    def execute(self, *,
                dregs : list[data_reg]|None = None,
                gregs : list[greg_base]|None = None,
                dts : dict[str,adt]|None = None,
                modifiers : set[mod],
                operand_modifiers : dict[str,set[opd_mod]],
                **kwargs) -> str:
        """
        Performs checks on all arguments, generates the parameters for the underlying
        implementation and calls it

        :param dregs: Data registers to use; may be empty for operations
            without data operands
        :param gregs: GP registers to use
        :param dts: Data type to use for each named operand
        :param modifiers: operation modifiers
        """

        dregs = dregs if dregs else []
        gregs = gregs if gregs else []
        dts = dts if dts else {}

        resolved_operands = kwargs.copy()

        resolved_operands |= self.resolve_dregs(dregs, dts)
        resolved_operands |= self.resolve_gregs(gregs)


        sigs = self.get_signatures()
        matched_sig = next(
            (s for s in sigs if \
                s.match_intent(modifiers=modifiers,
                               operand_modifiers=operand_modifiers,
                               kwargs=resolved_operands,
                               dts=dts)),
            None)
        if not matched_sig:
            self.diagnose_failure(modifiers, operand_modifiers, kwargs, dts)
            self._auto_diagnose(modifiers, operand_modifiers, kwargs, dts)

            regtypes = [f"{name}:{type(r).__name__}" for name,r in resolved_operands.items() \
                    if isinstance(r, (data_reg,greg_base,mreg_base))]

            regtypestr = ", ".join(regtypes)

            dtstr = ", ".join(f"{name}:{dt.name}" for name,dt in dts.items())


            itaddition = f"\n  it: {kwargs['it'].name}" if 'it' in kwargs else ""

            opd_mod_str = ""
            for name,modset in operand_modifiers.items():
                modsetstr = ", ".join(f"{m.name}" for m in modset)
                opd_mod_str += f"    {name}:{modsetstr}\n"
            raise ValueError(
                f"Invalid configuration for {type(self).__name__}.\n"
                f"  Modifiers: {modifiers}\n"
                f"  operand_modifiers:\n"
                f"{opd_mod_str}"
                f"  dts: {dtstr}\n"
                f"  reg types: {regtypestr}{itaddition}"
            )


        matched_sig.validate_allocation(resolved_operands)

        resolved_operands['modifiers'] = modifiers
        resolved_operands['operand_modifiers'] = operand_modifiers
        resolved_operands['dts'] = dts

        # opmem has a different interface
        resolved_operands['dregs'] = dregs

        return self.implementation(**resolved_operands)


    @abstractmethod
    def implementation(self, **kwargs) -> str:
        """
        Actual implementation of the operation to be implemented
        by the inheriting class
        """
        raise NotImplementedError(self.NIE_MESSAGE)
