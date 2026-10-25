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

#: what a call that doesn't fit raises - argument checks, the signatures,
#: their constraints, implementations refusing what their signatures allow
_REFUSALS = (ValueError, TypeError, KeyError, IndexError, NotImplementedError)


class operation(ABC):
    """
    Abstraction over operations/instructions
    """

    NIE_MESSAGE="Inheriting class must implement this method"

    #: optional ISA extensions enabled - signatures requiring others are
    #: hidden; None: no gating, every signature
    enabled_features : frozenset | None = None

    def get_signatures(self) -> list[operation_signature]:
        """
        The operation signatures this operation supports: those whose
        required features are on (every one while nothing gates them)

        :return: List of valid signatures
        """
        signatures = self.all_signatures()
        if self.enabled_features is None:
            return signatures
        return [s for s in signatures if (s.requires or frozenset()) <= self.enabled_features]

    @abstractmethod
    def all_signatures(self) -> list[operation_signature]:
        """
        Every signature of this operation, whatever features are on: what
        each operation defines, get_signatures filtering it

        :return: List of signatures
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

        # fits(): the signature the call takes, None where it would raise -
        # every check and the implementation run, no text kept
        query = kwargs.pop("_query", False)
        dregs = dregs if dregs else []
        gregs = gregs if gregs else []
        dts = dts if dts else {}

        resolved_operands = kwargs.copy()

        try:
            resolved_operands |= self.resolve_dregs(dregs, dts)
            resolved_operands |= self.resolve_gregs(gregs)
        except _REFUSALS:
            if query:
                return None
            raise


        sigs = self.get_signatures()
        matched_sig = next(
            (s for s in sigs if \
                s.match_intent(modifiers=modifiers,
                               operand_modifiers=operand_modifiers,
                               kwargs=resolved_operands,
                               dts=dts)),
            None)
        if not matched_sig and query:
            return None
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


        try:
            matched_sig.validate_allocation(resolved_operands)
        except _REFUSALS:
            if query:
                return None
            raise

        resolved_operands['modifiers'] = modifiers
        resolved_operands['operand_modifiers'] = operand_modifiers
        resolved_operands['dts'] = dts

        # opmem has a different interface
        resolved_operands['dregs'] = dregs

        if not query:
            return self.implementation(**resolved_operands)
        try:
            self.implementation(**resolved_operands)
        except _REFUSALS:
            return None
        return matched_sig

    def fits(self, **kwargs) -> operation_signature | None:
        """
        Whether a call with these arguments - the operation's own - would
        work: the signature it would take, None where it would raise. Nothing
        is raised and no text kept; a caller asks instead of trying.

        :return: the signature, or None
        """
        try:
            return self(**kwargs, _query=True)
        except _REFUSALS:
            return None

    def forms(self, *, dts : dict | None = None, rtypes : dict | None = None,
              modifiers : set | None = None,
              operand_modifiers : dict | None = None) -> list[operation_signature]:
        """
        The signatures whose named operands have these data types and
        register types, and that carry these modifiers (at least): which
        forms there are for given operands, without registers or values -
        the FMA's with a scalar operand, whether loads take a predicate

        :param dts: operand name: data type
        :param rtypes: operand name: register type
        :param modifiers: modifiers each form has, among its own
        :param operand_modifiers: operand name: modifiers it has, among its own
        :return: the signatures, in the operation's order
        """
        return [s for s in self.get_signatures()
                if s.offers(dts=dts, rtypes=rtypes, modifiers=modifiers,
                            operand_modifiers=operand_modifiers)]


    @abstractmethod
    def implementation(self, **kwargs) -> str:
        """
        Actual implementation of the operation to be implemented
        by the inheriting class
        """
        raise NotImplementedError(self.NIE_MESSAGE)
