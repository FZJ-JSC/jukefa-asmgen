# --------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# --------------------------------------------------------------------
"""
The values an operand may take under its constraints
"""

from .constraint import (
    ArgumentDependencyError, ConstraintDoesNotApplyError,
    operand_constraint, minmax_constraint, oneof_constraint, multiple_constraint,
    either_constraint, otherplusn_constraint, otherplusnmod_constraint,
    sameval_constraint, differentval_constraint,
)


class unbounded_domain_error(ValueError):
    """
    A domain still unbounded where a constraint needs its values listed
    """


class constraint_domain:
    """
    The integer values - register indices, immediates - an operand may take
    under its constraints, as a sorted list.

    The constraints with finitely many solutions narrow the domain first, from
    unbounded (or from universe, a register file's indices) to a list:
    minmax_constraint, oneof_constraint, either_constraint where every
    option is finite (the union of the options), and the ones fixing the
    value from another operand's (otherplusn, otherplusnmod, sameval). Then
    the ones with infinitely many filter the list: multiple_constraint,
    differentval_constraint. Then the ones
    without a handler: called on each value left, made as the operand holds
    it (makeval). A domain still unbounded when a filter or an unknown
    constraint needs its values is an error (unbounded_domain_error).

    A constraint that doesn't apply under the modifiers is skipped; one that
    depends on another operand needs it in context (ArgumentDependencyError
    otherwise). Parameters are specialized by the modifiers as the
    constraints' own checks do.

    :param constraints: the operand's value constraints
    :param name: the operand's name, for the constraints and messages
    :param modifiers: the operation's modifiers, specializing parameters
    :param context: other operands' values, for constraints depending on them
    :param universe: the values to start from (a register file's indices);
                     None - unbounded
    """

    def __init__(self, constraints, *, name : str = "value", modifiers=frozenset(),
                 context : dict | None = None, universe=None):
        self.name = name
        self.modifiers = set(modifiers)
        self.context = dict(context or {})
        self.values : list[int] | None = None if universe is None else sorted(set(universe))
        finite, infinite, unknown = [], [], []
        for c in constraints or []:
            kind = self._kind(c)
            (finite if "finite" == kind else infinite if "infinite" == kind else unknown).append(c)
        for c in finite:
            self._narrow(self._finite_values(c))
        for c in infinite:
            self._filter_multiple(c)
        for c in unknown:
            self._filter_call(c)

    # --- classification ------------------------------------------------------

    DEPENDENT_FINITE = (otherplusn_constraint, otherplusnmod_constraint, sameval_constraint)

    def _kind(self, c) -> str:
        if isinstance(c, (minmax_constraint, oneof_constraint) + self.DEPENDENT_FINITE):
            return "finite"
        if isinstance(c, (multiple_constraint, differentval_constraint)):
            return "infinite"
        if isinstance(c, either_constraint) and all(
                any("finite" == self._kind(o) for o in option) for option in c.options):
            return "finite"
        return "unknown"

    def _params(self, c) -> dict | None:
        """
        The constraint's parameters as its check would use them; None where it
        doesn't apply under the modifiers
        """
        params = dict(getattr(c, "params", {}))
        if self.modifiers:
            try:
                c.specialize_params(name=self.name, modifiers=self.modifiers,
                                    context=self.context, params=params)
            except ConstraintDoesNotApplyError:
                return None
        return params

    # --- handlers ------------------------------------------------------------

    def _finite_values(self, c) -> set[int] | None:
        if isinstance(c, either_constraint):
            out = set()
            for option in c.options:
                sub = constraint_domain(option, name=self.name, modifiers=self.modifiers,
                                        context=self.context, universe=self.values)
                if sub.values is None:
                    raise unbounded_domain_error(
                        f"{self.name}: an option of {type(c).__name__} is unbounded")
                out |= set(sub.values)
            return out
        params = self._params(c)
        if params is None:
            return None
        if isinstance(c, minmax_constraint):
            return set(range(params["minval"], params["maxval"] + 1))
        if isinstance(c, self.DEPENDENT_FINITE):
            # one value, given the other operand's
            other = self._other(c, params)
            if isinstance(c, sameval_constraint):
                return {other}
            value = other + params["offset"]
            return {value % params["modval"] if isinstance(c, otherplusnmod_constraint) else value}
        return set(params["valset"])

    def _other(self, c, params) -> int:
        other = params["other"]
        if other not in self.context:
            raise ArgumentDependencyError(name=self.name, deps=[other])
        return c.getint(self.context[other])

    def _narrow(self, values : set[int] | None):
        if values is None:
            return
        self.values = sorted(values if self.values is None else values & set(self.values))

    def _need_bounded(self, c):
        if self.values is None:
            raise unbounded_domain_error(
                f"{self.name}: {type(c).__name__} needs a bounded domain - no finite "
                f"constraint or universe bounds it")

    def _filter_multiple(self, c):
        params = self._params(c)
        if params is None:
            return
        if isinstance(c, differentval_constraint):
            # every value but the other operand's
            other = self._other(c, params)
            self._need_bounded(c)
            self.values = [v for v in self.values if v != other]
            return
        self._need_bounded(c)
        self.values = [v for v in self.values if 0 == v % params["multiple"]]

    def _filter_call(self, c : operand_constraint):
        self._need_bounded(c)
        make = getattr(c, "makeval", lambda v: v)
        kept = []
        for v in self.values:
            try:
                c(self.name, self.modifiers, make(v), self.context)
            except ConstraintDoesNotApplyError:
                kept.append(v)
            except ArgumentDependencyError:
                raise
            except ValueError:
                continue
            else:
                kept.append(v)
        self.values = kept

    # --- queries -------------------------------------------------------------

    @property
    def bounded(self) -> bool:
        return self.values is not None

    def __contains__(self, value : int) -> bool:
        return self.values is None or value in self.values

    def single(self) -> int | None:
        """
        The one value the operand can take, if there is exactly one
        """
        return self.values[0] if self.values is not None and 1 == len(self.values) else None
