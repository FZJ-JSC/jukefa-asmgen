# asmgen 0.6: operations instead of one-purpose methods

The generator grew from one-purpose methods (`load_vector_immoff`,
`add_greg_imm`, `zero_vreg`, ...) towards signatured operations (`load`,
`store`, `move`, `fma`, ...) whose signatures say what an ISA can do. 0.6
finishes the move: what the old methods did becomes operations with
signatures and constraints, the old methods go, and the workarounds callers
built around missing features go with them. This file is the plan and the
log; it is updated as the work goes.

Scope: asmgen (`plannable_operations`, 0.6.0), and the callers that have to
follow - ukrgen (`pack`) and ukropt (`master`).

## Principles

1. Each phase keeps the generated assembly byte-identical, except where the
   phase says otherwise - checked by golden tests (phase 0).
2. Old methods first become thin deprecated wrappers over the new
   operations, so ukrgen and ukropt keep working; they are deleted only once
   their callers have moved (phase 9).
3. Constraints live in signatures: what an ISA can encode - immediate
   ranges, offsets, shifts, lanes - is an operand constraint, and "can this
   ISA do X" is a question to the signatures, not a call that may raise.

## Audit (asmgen c5a9324, ukrgen da8ee04, ukropt fee96f1)

### What asmgen offers

- The base generator: 100 methods, 80 of them abstract (each backend writes
  its own text), and 13 properties.
- Signatured operations: `load`, `store`, `move`, `fma`, `fadd`, `fmul`, and
  SME's `fopa`. rvv071 inherits rvv's, but they don't use its `dt_suffixes`:
  it effectively runs on its legacy methods.
- Scalar delegation in one place: `rvv_opmem` holds a `riscv64_opmem`, merges
  its signatures and forwards scalar registers to it - the model for the
  scalar operations below.

### Who uses the old methods

| family | members (calls outside the backends) | callers |
|---|---|---|
| address arithmetic | `add_greg_greg` (14), `add_greg_imm` (20), `mul_greg_greg` (6), `mul_greg_imm` (3), `shift_greg_left` (7), `add_greg_voff` (2), `{add,sub}_greg_greg[_shifted]` by constructed name | ukrgen's emitters, asmgen `callconv`, ukropt |
| moves | `mov_greg` (17), `mov_greg_imm` (12), `mov_greg_to_freg` (1) | emitters, ukropt |
| zeroing | `zero_vreg`, `zero_treg`, `zero_freg`, `zero_greg` | emitters, ukropt |
| scalar memory | `load_greg`, `load_freg`; `load_{type}`/`store_{type}` by constructed name | signature emitter, asmgen `fngen`, `callconv` |
| vector memory | `load_vector_gather`, `_gregstride`, `_lane`; the `load/store_vector*`, `*_scalar*_immoff`, `*_tile*` families by constructed name | ukrgen's old emitter and specializer |
| vector length, masks | `vlen_to_greg`, `simd_size_to_greg`, `mask_all_true`, `greg_to_voffs`, `indexable_elements`, `simd_size`, `is_vla`, `c_simd_size_function` | emitters, BLIS C generation, ukropt |
| limits | `max_load_immoff`, `max_fload_immoff`, `max_bcast_immoff`, `max_load_voff`, `max_add_voff` | ukrgen's old path only |
| control (stays) | `cb`, `jump`, `jzero`, `label`, loops, `kiterkleft`, `ret` | emitters, ukropt |

Unused outside the backends under any name: `fill_vector`, `load_pointer`,
`mov_freg`, `mov_greg_to_param`, `mov_param_to_greg[_shift]`,
`prefetch_l1_immoff`, `shift_greg_right`, `jfzero`, `jvzero`,
`loopbegin_nz`, `labelstr`, all `min_*` limits, `max/min_prefetch_offset`.

### Workarounds callers built

ukrgen's signature emitter:
- function framing as text (`.text`, `.global`, the GNU-stack note) and
  `.balign` for loop heads;
- the vector length from `vlen_to_greg`, which only SVE and SME implement:
  on RVV, NEON and x86 the emitter raises, although `simd_size_to_greg` does
  the same everywhere;
- capability probes: `{op}_greg_greg_shifted` called with dummy registers to
  see whether it raises, `hasattr(gen, "mask_all_true")`, `try` around
  `greg_to_voffs`, `move`, `vlen_to_greg`;
- register capacity found by reserving until an exception;
- `get_reg_tracker` optional for older generators, `hasattr(gen,
  "set_features")` in `isa_map`;
- IR operations dispatched by name, `getattr(gen, op.op)`: the IR's op names
  are the generator's attribute names (`fmul`, `fadd` in the presets).

ukropt (pairs, fmaload):
- immediates split by hand (RISC-V's `addi` range; AArch64's `mov` only
  encodes some constants) - asmgen emits any immediate unchecked;
- nop encodings, vector bytes and FP16 `-march` flags by ISA name;
- the FMA's scalar-operand form and SVE's predicates found by trial;
- free registers counted by reserving until failure;
- `c_simd_size_function` patched from `inline` to `static inline` (C).

### Duplications

- `simd_size_to_greg` writes elements of a type (`mov x5,#2` on NEON in FP64)
  - what `vlen_to_greg` does on SVE (`cntd`): one meaning, two names, on
  disjoint backends. `simd_size` is bytes known at compile time (1 on VLA);
  `c_simd_size_function` bytes at run time; `indexable_elements` a lane
  range - an operand constraint.
- The limits (`max/min_*_immoff`, `*_voff`, prefetch) against signature
  constraints, which the signature emitter already reads.
- Scalar memory three ways (`load_scalar_immoff`, `load_freg`,
  `load([freg], IOFFSET)`); broadcast loads two (`load_vector_bcast1*`,
  `load` with `BCAST`).
- Constants: `zero_greg` and `mov_greg_imm(0)`; `zero_vreg` and
  `fill_vector`.
- Framing and alignment three times: ukrgen's emitter, ukropt's pairs,
  asmgen's `fngen`/`asmwrap`/`cppgen`.

## Decisions

1. ukrgen's old mechanism - the old emitter (`emitters/asm.py`), specializer
   (`specializers/asm.py`), the old model and what only they use - moves to a
   `legacy/` directory as a reference (as v1 methods were for the preload
   work), and is removed before ukrgen 1.0.
2. rvv071 is ported: its operations print 0.7.1's forms; FP32 and FP64
   kernels must build. Loads and stores at another element width than the
   one `vsetvli` set (1.0's mixed precision) aren't a goal.
3. Immediates are operands - `operand_shape` with `operand_type.IMMEDIATE` -
   their ranges value constraints.
4. `set(CONST, value)` is the one place constants are made, `mov_greg_imm`
   included (materializing sequences: `li`, `movz`/`movk`, `movabs`);
   `move` is register to register.
5. `set(VLEN)`: without `dt` the vector length in bytes, with `dt` in
   elements of that type.
6. Deprecated wrappers warn for one tagged release of asmgen, then go.

Also agreed:
- x86 offsets and address immediates keep today's 255-byte limit by
  default; the full signed 32 bits only with
  `set_parameter(name="allow-32bit-offsets", value="on")` - larger offsets
  slow many AVX2/AVX-512 cores down and sped none up in our measurements.
- `c_simd_size_function` stays, built on `set(VLEN)` in its inline asm, and
  emits `static inline`.
- `simd_size` is compile-time knowledge: it raises on VLA instead of 1.
- `gen.nop()` gives one nop; padding to an exact offset stays the caller's
  (ukropt knows each ISA's nop size).
- Framing - section, `.global`, alignment, the GNU-stack note - and
  loop-head alignment belong to `callconv`/`fngen`.
- `-march` and friends: `compilation/compiler_presets.py` improved, or moved
  into the generators as properties following the enabled features (`ZVFH`
  on: `_zfh_zvfh`); `emulation.py`, `compilation/tools.py` and
  `tests/test_asm_correct.py` use it today.
- `examples/axpy.py` keeps working; `examples/rvv_lmul.py` (broken now) is
  fixed along the way.

## Phases

Each phase lists its steps and when it is done. Status: [ ] open, [~] in
progress, [x] done.

### Phase 0: a safety net [x]

- [x] Golden snapshots (`tests/golden`, `tests/test_golden.py`): every
      base method as `test_asm_implemented` calls it, those with a data type
      in FP64, FP32 and FP16, those with an immediate over values around
      encodings' limits; every signature of every operation, instantiated to
      satisfy its constraints (ranges, value sets, multiples, another operand
      plus an offset), immediates at both ends of their ranges. Keys by
      content, not position; errors recorded as behavior; the same under
      every hash seed. About 13,000 entries over the eight backends, 30 s.
      `python -m tests.golden record [BACKEND]` after an intended change,
      `python -m tests.golden diff` to see one.
- [x] The snapshots assembled by each backend's assembler (rvv071: text
      only): the entries that don't assemble today are the baseline, an
      entry that stops assembling fails.
- [x] `examples/axpy.py` run as a test (`tests/test_examples.py`): every
      ISA it offers, both arithmetic forms, unrolled once and three times,
      against a recording, its output assembled. It was broken - it built
      its register tracker from `gen.max_*regs`, which the backends
      implement as methods - and now takes `gen.get_reg_tracker()`.
- [x] ukrgen: snapshots of the family pipeline's kernels
      (`tests/flow/golden_kernels`): five configurations' families at their
      first sizes (RVV, NEON, SVE, AVX2, AVX-512) and the K1's 2v x 14.
- [x] ukropt: `pairs` and `fmaload` built for RVV, NEON, SVE, x86
      (`tests/test_builds.py`), where the compilers are there.

Done when: the snapshots pass on the unchanged code, and a deliberate
change to one method's output fails them. (Checked: RVV's `zero_vreg`
changed fails exactly its three entries.)

Found by the snapshots, for the phases to come:
- SVE and SME advertise BF16 forms of `fadd`, `fmul` and some moves that
  raise `KeyError` (no BF16 suffix): 30 and 40 signatures.
- NEON's integer widening `fma` with `PART` raises "Unsupported datatype":
  16 signatures.
- SME's tile numbers per data type aren't constraints: 48 signatures can
  only be instantiated by guessing.
- 11 RVV and 11 SVE `move` forms are rejected by their own signatures
  ("Invalid configuration").
- `max_gregs`, `max_vregs`, `max_fregs`, `max_mregs`, `max_tregs` are
  properties in the base class and methods in the backends (and
  `get_reg_tracker` calls them): one of the two, phase 7.
- Entries that don't assemble (fma128 346, fma256 286, avx512 289, neon
  262, sve 312, sme 549, rvv 193): partly immediates swept past what
  encodes, inline-only methods and label references - by design - partly
  to triage. The assemblers take the extensions this binutils knows (RVV
  without the BF16 vector extensions, SME with SME2: axpy's SME code uses
  predicate-as-counter registers).

### Phase 1: `fadd` → `add`, `fmul` → `mul` [x]

- [x] Operations and classes renamed: `{avx,neon,rvv,sve}/opd3/fadd.py` and
      `fmul.py` became `add.py` and `mul.py`, `rvv_fadd` became `rvv_add`
      and so on, the generators' attributes `add` and `mul`. Docstrings and
      messages name the operations; the instructions' mnemonics stay.
- [x] Deprecated for one release: `gen.fadd` and `gen.fmul` (properties on
      the base class, warning), the old class names (the `opd3` packages'
      module `__getattr__`, warning), axpy's `--arith fmul_fadd` (now
      `mul_add`). `tests/test_renamed.py` checks them.
- [x] ukrgen: op names follow (its 0157) - the packing configurations'
      `pack-tileop`, the presets, `op_support_builder.ast_op_map`, the
      scheduler's op keys (`fmul` kept there as an alias). Configurations
      still saying `fmul(...)` keep working through the aliases.

Checked: the snapshots, with keys `op fadd`/`op fmul` read as `op
add`/`op mul` and the classes' names in error messages as their new ones,
are identical on all eight backends and for axpy; ukrgen's packing kernels
(RVV, AVX-512, SVE) and golden kernels are identical. `test_asm_correct`'s
execution scenarios (ten on three backends) fail in this environment before
and after alike.

`fma` stays: fused multiply-accumulate.

### Phase 2: scalar arithmetic as operations [ ]

- [ ] `add`, `sub`, `mul` on general registers: register and register,
      register and immediate. Immediate ranges as constraints: RISC-V
      -2048..2047; AArch64 0..4095, optionally shifted by 12; x86 the
      255-byte default, 32-bit with `allow-32bit-offsets`. Where an ISA has
      no immediate form (RISC-V `mul`) the signature doesn't offer one.
- [ ] Shifted operands with an operation modifier `ISHIFT` and the shift as
      a constrained immediate (AArch64 `lsl #0..63`, x86 `lea` scales 1, 2,
      4, 8, RISC-V's `shNadd` with Zba): replaces
      `{add,sub}_greg_greg_shifted` and ukrgen's probe.
- [ ] Shifts of general registers left and right as an operation (name to
      settle here: `op/shift.py` is about lane and block moves).
- [ ] VL-scaled immediates for SVE's `addvl` (-32..31): replaces
      `add_greg_voff` and `max_add_voff`.
- [ ] Vector generators delegate scalar registers to their base ISA's
      operations, as `rvv_opmem` does.
- [ ] Wrappers: `add_greg_greg`, `add_greg_imm`, `sub_greg_greg`,
      `mul_greg_greg`, `mul_greg_imm`, `shift_greg_left/right`,
      `{add,sub}_greg_greg_shifted`, `add_greg_voff`.
- [ ] asmgen's own `callconv` (stack pointer) on the operations.

### Phase 3: moves [ ]

- [ ] `move([greg, greg])` (`mov_greg`), `move([freg, freg])` (`mov_freg`),
      `move([greg, freg], mods={BITS})` (`mov_greg_to_freg`).
- [ ] Parameters: `mov_greg_to_param`, `mov_param_to_greg[_shift]` as
      `load`/`store` with a `PARAM` modifier, inline output only.
- [ ] A broadcast from a register is a `move` with `BCAST`: replaces
      `fill_vector`.
- [ ] Wrappers for the above.

### Phase 4: the `set` operation [ ]

A new operation class for values that come from nowhere but the
instruction stream:

- [ ] `CONST`: `constant="zero"`, `"ones"`, or a value (general registers:
      the materializing sequence). Replaces `zero_greg`, `zero_freg`,
      `zero_vreg`, `zero_treg`, `mov_greg_imm`.
- [ ] `CONSTMASK`: `mask="alltrue"` and `"allfalse"`. Replaces
      `mask_all_true`.
- [ ] `VLEN`: bytes, or elements of `dt` where given - on every backend
      (RVV and rvv071 `vsetvli`/`csrr vlenb`, SVE/SME `cnt*`/`rdvl`,
      NEON/x86 constants). Replaces `simd_size_to_greg` and `vlen_to_greg`;
      ukrgen's vector length works on RVV, NEON and x86.
- [ ] `INDEX`: a vector of element index times a stride register. Replaces
      `greg_to_voffs`.
- [ ] `c_simd_size_function` on `set(VLEN)`, `static inline`.
- [ ] Wrappers for the above.

### Phase 5: memory [ ]

- [ ] Every old load and store mapped onto `load`/`store` signatures, the
      missing modifiers added: plain, `IOFFSET`, `GOFFSET`, VL-scaled
      offsets, `BCAST` (with offset and post-increment), strides (register,
      immediate), gather and scatter, lanes, tiles, and prefetch (from
      `riscv64/opmem/prefetch.py` to every ISA).
- [ ] Scalar loads and stores (`load_greg`, `load_freg`, `load_pointer`,
      `*_scalar_immoff`) delegated to the base ISA.
- [ ] `callconv`/`fngen` spills, fills and stack parameters on
      `load`/`store`, not constructed names.
- [ ] The limits become constraints: `max/min_load_immoff`,
      `max/min_fload_immoff`, `max/min_bcast_immoff`, `max/min_load_voff`,
      `max/min_prefetch_offset`.
- [ ] rvv071: its operations print 0.7.1's mnemonics and `vsetvli`, their
      signatures offer only the element width `vsetvli` set; FP32 and FP64
      kernels build.
- [ ] Wrappers for the old methods.

### Phase 6: framing and directives [ ]

- [ ] `fngen` writes the function: section, alignment, `.global`, label,
      body, return, and the file's GNU-stack note.
- [ ] Loop heads aligned through it (ukrgen's `loop-align`).
- [ ] `gen.nop()`.
- [ ] One implementation for standalone output; `asmwrap`/`cppgen` keep the
      inline one.

### Phase 7: questions instead of trials [ ]

- [ ] A query on each operation: which forms fit given operands and types
      (the FMA's scalar operand, SVE's predicates) - no call that may raise.
- [ ] `available(tag)` on the register tracker.
- [ ] Lane ranges as operand constraints: replaces `indexable_elements`.
- [ ] `simd_size` raises on VLA.
- [ ] `max_*regs`: properties or methods, the same in the base class and the
      backends.
- [ ] Compiler flags from the generator and its enabled features (or an
      improved `compiler_presets`), its users moved.

### Phase 8: the callers [ ]

ukrgen:
- [ ] The signature emitter on the operations: address arithmetic, zeroing,
      the vector length, masks, parameters, framing, loop alignment; its
      probes and compatibility shims gone; IR op names `add`/`mul`.
- [ ] The old mechanism into `legacy/`; the family pipeline's stages
      (`irmod`, `variant`) no longer import from it.

ukropt:
- [ ] `pairs` and `fmaload` without their workarounds: immediates through
      constraints and `set`, `gen.nop()` with ukropt's own nop sizes, vector
      bytes and compiler flags from the generator, forms by query, free
      registers from the tracker.

### Phase 9: removal [ ]

- [ ] The deprecated wrappers and the limits deleted.
- [ ] What the generator keeps: register constructors; control flow (`cb`,
      `jump`, `jzero`, `label`, loops, `kiterkleft`, `ret`); setup
      (`isaquirks`, `isaendquirks`, `isa_setup`, `isadata`,
      `create_callconv`, `get_reg_tracker`, parameters, features, output
      mode, `asmwrap`); host checks; `c_simd_size_function`; `nop()`.
- [ ] `examples/axpy.py` and `examples/rvv_lmul.py` work.

## Log

- 2026-10-23: audit and plan (asmgen c5a9324, ukrgen da8ee04, ukropt
  fee96f1).
- 2026-10-23: phase 0 - golden snapshots of methods and operations on all
  eight backends, with an assembly baseline.
- 2026-10-23: phase 0 done - axpy as a test (fixed), ukrgen's golden
  kernels (its 0156), ukropt's build checks (its 0010).
- 2026-10-23: phase 1 done - fadd/fmul renamed add/mul, the old names
  deprecated (asmgen 0037, ukrgen 0157).
