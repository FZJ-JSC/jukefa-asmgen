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


def _ranges(values) -> list[range]:
    """
    Sorted integers as a union of consecutive runs
    """
    out, run = [], None
    for v in sorted(set(values)):
        if run is not None and v == run[1]:
            run[1] = v + 1
        else:
            if run is not None:
                out.append(range(*run))
            run = [v, v + 1]
    if run is not None:
        out.append(range(*run))
    return out


def _cut(r : range, lo : int, hi : int) -> range:
    """
    The part of an ascending range within lo..hi-1
    """
    if lo <= r.start:
        start = r.start
    else:
        start = r.start + -(-(lo - r.start) // r.step) * r.step
    return range(start, max(start, min(r.stop, hi)), r.step)


def _intersect(a : list[range], b : list[range]) -> list[range]:
    out = []
    for x in a:
        for y in b:
            if 1 == x.step == y.step:
                r = range(max(x.start, y.start), min(x.stop, y.stop))
            elif 1 == y.step:
                r = _cut(x, y.start, y.stop)
            elif 1 == x.step:
                r = _cut(y, x.start, x.stop)
            else:
                small, other = (x, y) if len(x) <= len(y) else (y, x)
                out += _ranges(v for v in small if v in other)
                continue
            if len(r):
                out.append(r)
    return out


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
        #: the domain as a union of ranges; None - unbounded
        self.ranges : list[range] | None = None if universe is None else _ranges(universe)
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
        params = dict(c.params)
        if self.modifiers:
            try:
                c.specialize_params(name=self.name, modifiers=self.modifiers,
                                    context=self.context, params=params)
            except ConstraintDoesNotApplyError:
                return None
        return params

    # --- handlers ------------------------------------------------------------

    def _finite_values(self, c) -> list[range] | None:
        if isinstance(c, either_constraint):
            out = []
            for option in c.options:
                sub = constraint_domain(option, name=self.name, modifiers=self.modifiers,
                                        context=self.context)
                if sub.ranges is None:
                    raise unbounded_domain_error(
                        f"{self.name}: an option of {type(c).__name__} is unbounded")
                out += sub.ranges
            return out
        params = self._params(c)
        if params is None:
            return None
        if isinstance(c, minmax_constraint):
            return [range(params["minval"], params["maxval"] + 1)]
        if isinstance(c, self.DEPENDENT_FINITE):
            # one value, given the other operand's
            other = self._other(c, params)
            if isinstance(c, sameval_constraint):
                return [range(other, other + 1)]
            value = other + params["offset"]
            if isinstance(c, otherplusnmod_constraint):
                value %= params["modval"]
            return [range(value, value + 1)]
        return _ranges(params["valset"])

    def _other(self, c, params) -> int:
        other = params["other"]
        if other not in self.context:
            raise ArgumentDependencyError(name=self.name, deps=[other])
        return c.getint(self.context[other])

    def _narrow(self, ranges : list[range] | None):
        if ranges is None:
            return
        # a union of the options' ranges may overlap: kept as it is - in and
        # values ask each
        self.ranges = list(ranges) if self.ranges is None else _intersect(self.ranges, ranges)

    def _need_bounded(self, c):
        if self.ranges is None:
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
            self.ranges = [p for r in self.ranges for p in (_cut(r, r.start, other),
                                                          _cut(r, other + 1, r.stop)) if len(p)]
            return
        self._need_bounded(c)
        m = params["multiple"]
        out = []
        for r in self.ranges:
            if 0 == r.step % m and 0 == r.start % m:
                out.append(r)
            elif 1 == r.step:
                first = -(-r.start // m) * m
                if first < r.stop:
                    out.append(range(first, r.stop, m))
            else:
                out += _ranges(v for v in r if 0 == v % m)
        self.ranges = out

    def _filter_call(self, c : operand_constraint):
        self._need_bounded(c)
        make = c.makeval
        kept = []
        for v in self._enumerate(f"calling {type(c).__name__}"):
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
        self.ranges = _ranges(kept)

    #: the most values a domain lists, or a constraint without a handler is
    #: called on, one by one
    LIMIT = 1 << 16

    def _enumerate(self, why : str):
        if self.size > self.LIMIT:
            raise unbounded_domain_error(f"{self.name}: {self.size} values are too many for {why}")
        return sorted({v for r in self.ranges for v in r})

    # --- queries -------------------------------------------------------------

    @property
    def bounded(self) -> bool:
        return self.ranges is not None

    @property
    def size(self) -> int | None:
        """
        How many values (an upper bound where ranges overlap); None unbounded
        """
        return None if self.ranges is None else sum(len(r) for r in self.ranges)

    @property
    def values(self) -> list[int] | None:
        """
        The values, sorted - listed where there are few enough; None unbounded
        """
        return None if self.ranges is None else self._enumerate("listing them")

    def __contains__(self, value : int) -> bool:
        return self.ranges is None or any(value in r for r in self.ranges)

    def single(self) -> int | None:
        """
        The one value the operand can take, if there is exactly one
        """
        if self.ranges is None:
            return None
        found = {v for r in self.ranges if len(r) <= 1 for v in r}
        if any(len(r) > 1 for r in self.ranges) or 1 != len(found):
            return None
        return found.pop()
