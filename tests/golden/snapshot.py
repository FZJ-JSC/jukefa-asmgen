"""
Golden snapshots: what every generator method and every operation signature
emits on every backend, recorded so a change shows. The 0.6 rework
(docs/V06_REWORK.md) keeps this output byte-identical except where a phase
says otherwise.

An entry is a key - the method and its arguments, or the operation, its
modifiers and operand shapes - and what the call returned, or the error it
raised (raising is behavior too). Operations are called once per signature
with operands built to satisfy it, a second time at the far end of an
immediate's range where it has one.

    python -m tests.golden record [BACKEND ...]   the snapshots anew
    python -m tests.golden diff [BACKEND ...]     what changed
"""
from __future__ import annotations

import inspect
import os
import re
from itertools import count

from asmgen.registers import asm_data_type as adt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
OPERATIONS = ("load", "store", "prefetch", "move", "fma", "add", "sub", "mul", "shift", "set", "fopa")
DTS = (adt.FP64, adt.FP32, adt.FP16)
IMMEDIATES = (0, 1, 8, 255, 256, 2047, 2048, 4096, -8, -2048, -2049)
IMM_PARAMS = ("imm", "offset", "ioffset", "shift")


def backends() -> dict:
    """
    The generators snapshotted, by name: constructors
    """
    from asmgen.asmblocks.avx import fma128, fma256, avx512
    from asmgen.asmblocks.neon import neon
    from asmgen.asmblocks.sve import sve
    from asmgen.asmblocks.sme import sme
    from asmgen.asmblocks.rvv import rvv
    from asmgen.asmblocks.rvv071 import rvv071
    return {"fma128": fma128, "fma256": fma256, "avx512": avx512, "neon": neon,
            "sve": sve, "sme": sme, "rvv": rvv, "rvv071": rvv071}


def make(name: str):
    """
    A generator for standalone text, every optional feature on: every
    signature reachable
    """
    gen = backends()[name]()
    gen.set_output_inline(yesno=False)
    feats = getattr(gen, "available_features", None)
    if callable(feats) and hasattr(gen, "set_features"):
        try:
            gen.set_features(list(feats()))
        except Exception:  # pylint: disable=broad-except
            pass
    return gen


def _plain(value) -> str:
    """
    A value as text without what varies between runs: registers by their
    names (their __str__), other objects without their addresses
    """
    text = str(value) if type(value).__str__ is not object.__str__ else repr(value)
    return re.sub(r" at 0x[0-9a-f]+", "", text)


def _text(value) -> str:
    text = value if isinstance(value, str) else _plain(value)
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _error(err: Exception) -> str:
    msg = str(err).splitlines()[0] if str(err) else ""
    msg = re.sub(r" at 0x[0-9a-f]+", "", msg)
    return f"!{type(err).__name__}: {msg[:160]}"


def _arg(value) -> str:
    if isinstance(value, adt):
        return value.name
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_arg(v) for v in value)) + "}"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_arg(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_arg(v) for v in value) + "]"
    if isinstance(getattr(value, "name", None), str) and not isinstance(value, (str, int)):
        return value.name
    return _plain(value)


def _call(fn, kwargs: dict) -> str:
    try:
        return _text(fn(**kwargs)) if callable(fn) else _text(fn)
    except Exception as err:  # pylint: disable=broad-except
        return _error(err)


def method_entries(name: str) -> dict:
    """
    Every base method the implementation test knows, as it calls them; those
    with a data type in FP64, FP32, FP16; those with an immediate over
    values around encodings' limits
    """
    from tests.test_asm_implemented import function_tests
    out = {}
    for mname, args in function_tests:
        gen = make(name)
        fn = getattr(gen, mname, None)
        if fn is None:
            out[f"method {mname}"] = "!missing"
            continue
        try:
            built = {k: v(gen) for k, v in (args or {}).items()}
        except Exception as err:  # pylint: disable=broad-except
            out[f"method {mname}"] = "!arguments " + _error(err)[1:]
            continue
        variants = [built]
        if "dt" in built:
            variants += [dict(built, dt=d) for d in DTS if d != built["dt"]]
        for p in IMM_PARAMS:
            if isinstance(built.get(p), int) and not isinstance(built.get(p), bool):
                variants += [dict(built, **{p: v}) for v in IMMEDIATES if v != built[p]]
        for kwargs in variants:
            key = f"method {mname}(" + ", ".join(f"{k}={_arg(v)}" for k, v in sorted(kwargs.items())) + ")"
            out[key] = _call(fn, kwargs)
    return out


def _register(gen, rtype: str, idx: int, dt):
    if "GP" == rtype:
        return gen.greg(idx)
    if "FP" == rtype:
        return gen.freg(idx, dt)
    if "VEC" == rtype:
        return gen.vreg(idx)
    if "MASK" == rtype:
        return gen.mreg(idx)
    if "TILE" == rtype:
        params = inspect.signature(gen.treg).parameters
        return gen.treg(idx, dt) if "dt" in params else gen.treg(idx)
    raise ValueError(f"no register type {rtype}")


START = {"GP": 10, "FP": 1, "VEC": 1, "MASK": 1, "TILE": 0}


def _shape_key(name, shape) -> str:
    mods = ",".join(sorted(m.name for m in (shape.modifiers or ())))
    rtype = shape.rtype.name if shape.rtype is not None else "-"
    dt = shape.dt.name if shape.dt is not None else "-"
    return f"{name}:{shape.otype.name}:{rtype}:{dt}" + (f"{{{mods}}}" if mods else "")


def _params(shape) -> list[dict]:
    return [getattr(c, "params", None) or {} for c in (shape.value_constraints or ())]


def _minmax(shape):
    for p in _params(shape):
        if "minval" in p and "maxval" in p:
            return p["minval"], p["maxval"]
        for alternative in p.get("alternatives", ()):
            for c in alternative:
                q = getattr(c, "params", {}) or {}
                if "minval" in q and "maxval" in q:
                    return q["minval"], q["maxval"]
    return None


def _choose(shape, start: int, values: dict, far: bool = False):
    """
    A value for an operand satisfying its constraints, from start: None
    while an operand it depends on has none yet
    """
    val, low, high = start, None, None
    for p in _params(shape):
        if "other" in p and "offset" in p:
            if p["other"] not in values:
                return None
            val = values[p["other"]] + p["offset"]
            mod = next((p[k] for k in ("mod", "modulo", "n") if isinstance(p.get(k), int) and p.get(k)), None)
            return val % mod if mod else val
    for p in _params(shape):
        if "valset" in p:
            vals = sorted(p["valset"])
            return vals[-1] if far else (vals[0] if start not in vals else start)
        if "minval" in p and "maxval" in p:
            low, high = p["minval"], p["maxval"]
    for p in _params(shape):
        if isinstance(p.get("multiple"), int) and p["multiple"] > 1:
            m = p["multiple"]
            val = -(-val // m) * m
    if low is not None and not low <= val <= high:
        val = high if far else low
    return val


def instantiate(gen, opname: str, sig, far: bool = False) -> dict | None:
    """
    Arguments calling gen.<opname> with operands satisfying sig: registers
    numbered per type, distinct, within their constraints (ranges, multiples,
    another operand plus an offset); immediates at the low end of their range
    (the high end with far). None where it can't be built
    """
    values, regs, imms = {}, {}, {}
    nexts = {k: v for k, v in START.items()}
    pending = list(sig.operands.items())
    for _ in range(len(pending) + 1):
        later = []
        for name, shape in pending:
            if "NAME" == shape.otype.name:
                imms[name] = values[name] = "p"
                continue
            if "IMMEDIATE" == shape.otype.name:
                mm = _minmax(shape)
                start = (mm[1] if far else mm[0]) if mm else 0
                val = _choose(shape, start, values, far)
                if val is None:
                    later.append((name, shape))
                    continue
                imms[name] = values[name] = val
                continue
            rtype = shape.rtype.name
            idx = _choose(shape, nexts[rtype], values)
            if idx is None:
                later.append((name, shape))
                continue
            nexts[rtype] = max(nexts[rtype], idx + 1)
            regs[name] = (_register(gen, rtype, idx, shape.dt), idx, shape)
            values[name] = idx
        pending = later
        if not pending:
            break
    if pending:
        return None
    data = sorted((n, r, s) for n, (r, _, s) in regs.items() if "DATA" == s.orole.name)
    kwargs = {}
    opmods = {n: set(s.modifiers) for n, (_, _, s) in regs.items() if s.modifiers}
    opmods |= {n: set(s.modifiers) for n, s in sig.operands.items() if n in imms and s.modifiers}
    if opmods:
        kwargs["operand_modifiers"] = opmods
    if sig.modifiers:
        kwargs["modifiers"] = set(sig.modifiers)
    kwargs.update(sig.structural_params or {})
    kwargs.update(imms)
    rest = {n: r for n, (r, _, s) in regs.items() if "DATA" != s.orole.name}
    if opname in ("load", "store", "prefetch"):
        kwargs["dregs"] = [r for _, r, _ in data]
        if data:
            kwargs["dt"] = data[0][2].dt
        addr = [n for n, (_, _, s) in regs.items() if "ADDRESS" == s.orole.name]
        if addr:
            kwargs["areg"] = rest.pop(addr[0])
    elif "set" == opname:
        for n, r, sh in data:
            kwargs[n] = r
            kwargs["dt"] = sh.dt
    elif "move" == opname:
        kwargs["dregs"] = [r for _, r, _ in data]
        kwargs["dts"] = [s.dt for _, _, s in data]
    else:
        for n, r, s in data:
            kwargs[n] = r
            if n.endswith("dreg"):
                kwargs[f"{n[0]}_dt"] = s.dt
    kwargs.update(rest)
    return kwargs


def operation_entries(name: str) -> dict:
    """
    Every signature of every operation the backend has, called with
    operands built to satisfy it
    """
    out = {}
    gen = make(name)
    for opname in OPERATIONS:
        op = getattr(gen, opname, None)
        if op is None or not hasattr(op, "get_signatures"):
            continue
        for sig in op.get_signatures():
            base = (f"op {opname} [" + ",".join(sorted(m.name for m in sig.modifiers)) + "] "
                    + " ".join(_shape_key(n, s) for n, s in sig.operands.items())
                    + (f" {_arg(sig.structural_params)}" if sig.structural_params else ""))
            tags = [("", False)]
            if any(_minmax(s) and _minmax(s)[0] != _minmax(s)[1] for s in sig.operands.values()
                   if "IMMEDIATE" == s.otype.name):
                tags.append((" @far", True))
            for tag, far in tags:
                key = base + tag
                n = 2
                while key in out:
                    key, n = f"{base}{tag} #{n}", n + 1
                try:
                    kwargs = instantiate(gen, opname, sig, far)
                except Exception as err:  # pylint: disable=broad-except
                    out[key] = "!uninstantiable " + _error(err)[1:]
                    continue
                if kwargs is None:
                    out[key] = "!uninstantiable"
                    continue
                out[key] = _call(op, kwargs)
    return out


def snapshot(name: str) -> dict:
    if "examples" == name:
        return examples_snapshot()
    out = method_entries(name)
    out.update(operation_entries(name))
    return out


def path(name: str) -> str:
    return os.path.join(DATA, f"{name}.txt")


def write(name: str, entries: dict):
    os.makedirs(DATA, exist_ok=True)
    with open(path(name), "w") as f:
        for key in sorted(entries):
            f.write(f"### {key}\n{entries[key]}\n")


def read(name: str) -> dict:
    out, key, lines = {}, None, []
    with open(path(name)) as f:
        for line in f.read().split("\n"):
            if line.startswith("### "):
                if key is not None:
                    out[key] = "\n".join(lines).rstrip("\n")
                key, lines = line[4:], []
            else:
                lines.append(line)
    if key is not None:
        out[key] = "\n".join(lines).rstrip("\n")
    return out


def compare(name: str) -> tuple[list, list, list]:
    """
    Against the recording: entries changed, gone, and new
    """
    old, new = read(name), snapshot(name)
    changed = [k for k in old if k in new and old[k] != new[k]]
    gone = [k for k in old if k not in new]
    added = [k for k in new if k not in old]
    return changed, gone, added


ASSEMBLERS = {
    # the assemblers themselves: compiler drivers check -march against their
    # own (older) lists - GCC 13's has no +sme
    "rvv": ["riscv64-linux-gnu-as", "-march=rv64gcv_zba_zfh_zvfh"],
    "neon": ["aarch64-linux-gnu-as", "-march=armv8.6-a+fp16+fp16fml+bf16+i8mm+dotprod"],
    "sve": ["aarch64-linux-gnu-as", "-march=armv9-a+sve2+bf16+i8mm+f64mm"],
    "sme": ["aarch64-linux-gnu-as", "-march=armv9-a+sme2+sme-f64f64+sme-i16i64+bf16"],
    "fma128": ["as"], "fma256": ["as"], "avx512": ["as"],
}


def assemble(name: str, entries: dict) -> list | None:
    """
    The entries that don't assemble: each emitting entry into one file for
    the backend's assembler, its errors mapped back to entries. None where
    no assembler is there (rvv071: none takes 0.7.1)
    """
    import shutil
    import subprocess
    import tempfile
    tool = ASSEMBLERS.get(name)
    if not tool or not shutil.which(tool[0]):
        return None
    lines, owner = [], []
    for key in sorted(entries):
        text = entries[key]
        if text.startswith("!") or not text.strip():
            continue
        for line in text.splitlines():
            lines.append(line)
            owner.append(key)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "s.s")
        with open(src, "w") as f:
            f.write("\n".join(lines) + "\n")
        r = subprocess.run([*tool, src, "-o", os.path.join(tmp, "s.o")],
                           capture_output=True, text=True)
    failing = set()
    for line in r.stderr.splitlines():
        m = re.match(r".*?s\.s:(\d+):", line)
        if m and 0 < int(m.group(1)) <= len(owner):
            failing.add(owner[int(m.group(1)) - 1])
    if r.returncode and not failing:
        # the assembler refused the whole file (its flags): nothing known
        print(f"{name}: the assembler failed outright: {r.stderr.strip()[:200]}")
        return None
    return sorted(failing)


def fails_path(name: str) -> str:
    return os.path.join(DATA, f"{name}.asm-fails.txt")


EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "examples")


def examples_snapshot() -> dict:
    """
    examples/axpy.py's output for every ISA it offers, both arithmetic
    forms, unrolled once and three times: the text, or how it failed
    """
    import importlib.util
    import subprocess
    import sys
    script = os.path.join(EXAMPLES, "axpy.py")
    spec = importlib.util.spec_from_file_location("axpy_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.join(os.path.dirname(EXAMPLES), "src"), os.environ.get("PYTHONPATH", "")]))
    out = {}
    for isa in module.ISA_MODULES:
        for arith in ("fma", "mul_add"):
            for unroll in (1, 3):
                r = subprocess.run([sys.executable, script, "--isa", isa, "--arith", arith,
                                    "--unroll", str(unroll)], capture_output=True, text=True, env=env)
                key = f"axpy --isa {isa} --arith {arith} --unroll {unroll}"
                out[key] = (_text(r.stdout) if 0 == r.returncode else
                            f"!exit {r.returncode}: " + (r.stderr.strip().splitlines() or [""])[-1][:160])
    return out


def example_isa(key: str) -> str:
    return key.split("--isa ")[1].split()[0]
