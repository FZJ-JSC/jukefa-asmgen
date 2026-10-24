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

## Conventions

- Asm-generating methods - operations' helpers too (`with_register`,
  `zero_greg`, ...) - take keyword arguments only, type-annotated, and
  return the text.
- An immediate in a register operand's place is named after it: `bdreg`'s
  is `bimm`, so `c = a << b` documents either form.
- Operation classes are named `op<kind>`: `opmem`, `opd3`, `opset`
  (`move` follows in phase 10, as `opmove`), each ISA's in
  `<isa>/op<kind>/{__init__,base,signatures}.py`.
- Features that would fit but nothing needs: docs/POSSIBLE_FEATURES.md.

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

### Phase 2: scalar arithmetic as operations [x]

- [x] `add`, `sub`, `mul` on general registers (`op/scalar.py`): register
      and register, register and immediate (`bimm`, by name - no `bdreg`), in
      `UINT64` and `SINT64`. Immediate ranges as constraints: RISC-V
      -2048..2047 (`sub`: -2047..2048, an `addi` negated); AArch64 up to
      4095, or a multiple of 4096 up to 4095 * 4096, either sign
      (`anyof_constraint`, new: one constraint of alternatives - execute
      takes the first signature matching by name, so alternatives can't be
      signatures); x86 signed 32 bits. `mul` has an immediate form on x86
      only.
      x86 (agreed): the 255-byte default with `allow-32bit-offsets` is for
      load and store offsets (phase 5) - arithmetic immediates take 32
      bits, as `add_greg_imm` always did. x86's two-operand forms now keep a
      destination that is the second source (`addq a,c`; `negq c` then
      `addq a,c` for `sub`), which the old methods overwrote.
- [x] Shifted operands: the operand modifier `ISHIFT` on `bdreg`, the
      amount an immediate `bdreg_shift` (as `ILANE` and `bdreg_lane`):
      c = a + (b << s). AArch64 `add`, `sub` with `LSL #0..63`; x86 `add`
      through `lea`, scales 1, 2, 4, 8 (0..3); RISC-V `add` through Zba's
      `sh1add`..`sh3add` (1..3), with the new RVV feature `ZBA` - off where
      features are set (ukrgen: `set_features([])`), on in a generator
      whose features never were, as every feature. Replaces
      `{add,sub}_greg_greg_shifted`, which raise `NotImplementedError`
      where no form fits as before: ukrgen's probe catches that.
- [x] Shifts as the operation `shift` with `opd3_modifier.SHL` or `SHR`
      (logical): `opd3`, as bit shifts change elements - lane shifts only
      move them, so they are `move`'s (`op/shift.py`). Immediate amounts
      (`bimm`) 0..63; register amounts on RISC-V and AArch64 (x86 takes
      them only in `%cl`: not offered).
- [x] VL-scaled immediates: `opd3_modifier.VOFFSET`, as opmem's - `bimm`
      counts vector lengths - on SVE (and SME): -32..31, `incb`/`decb`
      where the destination is the source and the count within 16 (as
      `add_greg_voff` wrote), `addvl` otherwise. A fixed vector length is
      a byte count: NEON's `add_greg_voff` adds `simd_size * offset`
      through `add`, its immediates checked; x86's through `add_greg_imm`.
      RVV has none (raises, as before; `set(VLEN)`, phase 4).
      `max_add_voff` stays until the old emitter leaves (phases 8, 9).
- [x] Vector generators delegate scalar registers to their base ISA's
      operations, as `rvv_opmem` does: the vector `opd3` bases (RVV,
      AArch64 SIMD, AVX) carry the scalar operation their class names
      (`scalar_class`, set by `add` and `mul`) as `scalar_op`, offer its
      signatures among their own, and hand calls with general registers
      to it first thing in `implementation`. `sub` is the scalar operation
      alone. `opd3.__call__` takes `bdreg` optional: with an immediate in
      its place the operands go by name.
- [x] Wrappers: `add_greg_greg`, `add_greg_imm`, `sub_greg_greg`,
      `mul_greg_greg`, `mul_greg_imm` (RISC-V's and AArch64's
      `mul_greg_imm` keep their two instructions), `shift_greg_left/right`,
      `{add,sub}_greg_greg_shifted`, `add_greg_voff` (SVE's 16 at a time).
- [x] asmgen's own `callconv` (stack pointer) on the operations.

Checked (step 2a): the snapshots changed in three entries only, RISC-V's
`add_greg_imm` with -2049, 2048 and 4096 - now an error, before text that
didn't assemble - besides the scalar signatures' new entries; ukrgen's
golden kernels and ukropt's builds unchanged.

Checked (step 2b): the snapshots changed only in entries that were errors,
or text that didn't assemble - AArch64's shifted add and sub with shifts
outside 0..63, 48 entries over NEON, SVE and SME, raise now - besides RISC-V's
shifted add with `ZBA` (the snapshots enable every feature) and the new
signatures' entries; ukrgen's golden kernels and ukropt's builds unchanged.
A debug print in `operation.execute`'s unsupported-modifier error is gone.

Checked (step 2c): the snapshots changed in NEON's `add_greg_voff` with
2047 and -2049 vector lengths only - text that didn't assemble - besides
SVE's and SME's new signatures; SVE's `incb` sequences unchanged; ukrgen's
golden kernels and ukropt's builds unchanged.

### Phase 3: moves [x]

- [x] `move([greg, greg])` (`mov_greg`), `move([freg, freg])` (`mov_freg`),
      `move([greg, freg], mods={BITS})` (`mov_greg_to_freg`): `scalar_move`
      (`op/scalar.py`) and the ISAs' (`riscv64_move`, `aarch64_move`,
      `x86_move`), carried by the vector moves as their `scalar_op` (SME's
      through SVE's move), calls whose registers are all scalar handed to
      it. `move_modifier.BITS` (new): UINT64 into DOUBLE, UINT32 into
      SINGLE, UINT16 into HALF (a conversion, `CAST`, is a possible
      feature). The ISAs' suffix tables are shared: RISC-V's in
      `riscv64/types.py`, x86's (`DT_SUFFIXES`, the AVX generators'
      `dt_suffixes`) in `x86/types.py`. The old methods' texts kept - x86's
      `vmovs{d,s,h}` with two unprefixed registers too, which doesn't
      assemble (a fix for later).
- [ ] Parameters: `mov_greg_to_param`, `mov_param_to_greg[_shift]` as
      `load`/`store` with a `PARAM` modifier, inline output only - with
      phase 5: they read and write memory (`ld`/`sd`, `ldr`/`str`, `movq`
      with `%[param]`), and no caller uses them.
- [x] A broadcast from a register is a `move` with `BCAST`: `fill_vector`
      is one method in the base class now, the backends' five gone. Its
      texts changed in their spacing, and SVE writes `mov z.d,d` (the
      alias of `dup z.d,z.d[0]`). x86's FP16 broadcast was `vbroadcastsh`,
      which isn't an instruction: it is `vpbroadcastw` now (AVX2 on xmm and
      ymm, AVX512BW on zmm), a new signature of the AVX move.
- [x] Wrappers for the above.

The parameter moves left for phase 5, phase 3 is done. Checked (step 3b):
the snapshots changed in `fill_vector`'s entries only, every one
assembling; ukrgen's golden kernels and ukropt's builds unchanged.

### Phase 4: the `set` operation [x]

A new operation class for values that come from nowhere but the
instruction stream:

- [~] `CONST`: `constant="zero"`, `"ones"`, or a value (general registers:
      the materializing sequence). Replaces `zero_greg`, `zero_freg`,
      `zero_vreg`, `zero_treg`, `mov_greg_imm`.
      Step 4a: the operation class `opset` (`op/opset.py`, as `opmem`
      and `opd3`; `gen.set(dreg=r, dt=dt,
      modifiers={opset_modifier.CONST}, constant="zero")` - the register
      set is `dreg`, its one output; the constant a structural parameter,
      so a signature names the constants it makes). `scalar_opset` and the
      ISAs' (`riscv64_opset`, `aarch64_opset`, `x86_opset`) for general
      and floating-point registers, carried by the vector ISAs'
      (`rvv_opset`, `neon_opset`, `sve_opset`, `avx_opset`; `sme_opset`
      adds tiles) as their `scalar_op` - each in `<isa>/opset/{__init__,
      base,signatures}.py`, as the other operations. RISC-V's FP suffix
      tables are `riscv64/types.py`'s now, the generator's and the
      operations' alike.
      Step 4b: `constant="one"` - 1 for integers, 1.0 for floating-point
      types, in dt's element size: general registers everywhere;
      floating-point ones on AArch64 (`fmov #1.0`) and x86; vectors on RVV
      (integers: `vmv.v.i`), NEON (`fmov #1.0`, `movi #1` up to 32 bits),
      SVE (`fdup #1.0`, `dup #1`) and x86 (all bits set - `vpcmpeqd`,
      `vpternlogd` with AVX-512 - shifted into place; `vpabsb` for
      bytes; checked by running them). Forms needing a scratch register
      aren't offered (docs/POSSIBLE_FEATURES.md). And `value=<int>` for
      general registers, any 64-bit value: RISC-V `li`; AArch64 `mov`
      where it encodes (`movz`, `movn`, a bitmask immediate), else `movz`
      and `movk` (checked under QEMU); x86 `movq` (the assembler encodes a
      value past 32 bits as `movabs`). `mov_greg_imm` on it, its texts
      kept. Zero done, `zero_greg`, `zero_freg`, `zero_vreg`,
      `zero_treg` on it, their texts kept; `"ones"` and values to come.
- [x] `CONSTMASK`: `mask="alltrue"` and `"allfalse"` (a structural
      parameter), on mask registers: SVE and SME `ptrue p.<T>` (`pn` too)
      and `pfalse p.b`; AVX-512 `kxnorw` and `kxorw`. `mask_all_true` on
      it. AVX2's masks are vector registers (its gathers' and scatters'):
      `vpcmpeqd` and `vpxor`, signatures of the AVX2 generators' set (the
      implementation tells a k register from a vector one by its type).
      AVX2's `mreg(i)` is `vreg(i)` - its gathers, masked loads and
      stores (`vmaskmov`, `vpmaskmov`) and variable blends take any vector
      register as a mask, none a general register - and the tracker
      shares the two pools (`max_mregs` is `max_vregs`), as NEON's floating-
      point and vector registers; AVX-512's are k registers, k0 reserved.
      AVX2's set has no k-register forms. Kept as it was: `kxnorw` sets 16 bits whatever the
      type, too few for FP16 and bytes in zmm (32 and 64 elements) - for
      `kxnord`/`kxnorq` later.
- [x] `VLEN`: `set(dreg=r, dt=dt, modifiers={VLEN})`, elements of dt -
      bytes where dt isn't given (UINT8, the default; the register is
      UINT64, the only case where set's dt isn't dreg's) - on
      every backend: RVV `vsetvli r, zero, e<bits>, m<LMUL>, ta, ma` (a
      register group's; it sets the vector type too) and rvv071's without
      the policies; SVE and SME `cnt{b,h,w,d}` (`simd_size_to_greg`'s
      `mov #0` and `inc` merged into it); NEON and x86 constants.
      `simd_size_to_greg` and `vlen_to_greg` are one implementation each in
      the base class, on it: every backend has both, ukrgen's vector length
      works on RVV, NEON and x86.
- [x] `STINDEX` (indices from a stride, where a `VINDEX` would hold any
      offsets): `set(dreg=v, dt=dt, modifiers={STINDEX}, streg=r)`, element i
      is i * r - RVV `vid.v` and `vmul.vx`; SVE and SME `index` (a w
      register below 8 bytes); AVX the stride broadcast times a table the
      generator keeps in its data (its `index_table` hands set the label; `rip` named through
      `rpref` like a register),
      2- to 8-byte elements (no byte multiply). NEON has none.
      `greg_to_voffs` is one implementation in the base class on it,
      raising `NotImplementedError` where there's no form, as ukrgen's
      emitter expects. AVX's 2-byte broadcast took `%cx`, which doesn't
      assemble: the source is r32 now. Kept as it was: on AVX2 the
      broadcast from a general register is AVX-512's (VL) - AVX2 has it
      only from xmm or memory.
- [x] `c_simd_size_function` on `set(VLEN)`, `static inline`: one
      implementation in the base class - a constant where the length is
      fixed, else `set(VLEN)` into the inline asm operand `%[byte_size]`
      (`inline_operand`); SME counts in streaming mode (`smstart`,
      `smstop` around it). ukropt's own `static` fix leaves it be (its
      0011).
- [x] Wrappers for the above: the `zero_*` methods, `mov_greg_imm`,
      `mask_all_true`, `simd_size_to_greg`, `vlen_to_greg`,
      `greg_to_voffs`.

Checked (step 4a): no snapshot entry changed, `set`'s new; ukrgen's golden
kernels - zeroing their accumulators through `zero_vreg` - and ukropt's
builds unchanged.

### Phase 5: memory [~]

- [ ] Every old load and store mapped onto `load`/`store` signatures, the
      missing modifiers added: plain, `IOFFSET`, `GOFFSET`, VL-scaled
      offsets, `BCAST` (with offset and post-increment), strides (register,
      immediate), gather and scatter, lanes, tiles, and prefetch (from
      `riscv64/opmem/prefetch.py` to every ISA).
- [x] Scalar loads and stores (`load_greg`, `store_greg`, `load_freg`,
      `store_freg`, `*_scalar_immoff`): one implementation each in the base
      class, on `load` and `store` - whose vector ISAs' carry the base
      ISA's scalar `opmem` already - `IOFFSET` where the offset isn't 0.
      Their texts are `opmem`'s now, which differ in format only (a space
      after commas; AArch64 writes `[sp]` for `[sp,#0]`): the snapshots
      and ukrgen's golden kernels are the same but for that (its 0158
      records them anew). Offsets past what encodes, text that didn't
      assemble, raise now (`opmem`'s constraints); AArch64's FP loads name
      the register as dt says (`ldr s0` for SINGLE), where the old methods
      took the register's own name whatever dt was. `load_pointer` goes
      with the parameters.
- [x] `callconv`/`fngen` spills, fills and stack parameters on
      `load`/`store` themselves (`IOFFSET` where the offset isn't 0), not
      constructed names.
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

### Phase 10: `opd3` and `move` renamed [ ]

With `bdreg` optional - an immediate in its place - and shifts among its
operations, `opd3` is the transforming operation: what changes elements,
where `move` only moves them. A name to say so, as ukrgen's
`lsc_transformation`: `optransform`, say. And `move`'s base class named as
the others (`opmem`, `opset`): `opmove`. After the other phases.

- [ ] `opd3` (class, module, modifier enum) renamed `optransform`, the old
      names deprecated for a release.
- [ ] `move` (class, module, modifier enum) renamed `opmove`, likewise.

## Log

- 2026-10-23: audit and plan (asmgen c5a9324, ukrgen da8ee04, ukropt
  fee96f1).
- 2026-10-23: phase 0 - golden snapshots of methods and operations on all
  eight backends, with an assembly baseline.
- 2026-10-23: phase 0 done - axpy as a test (fixed), ukrgen's golden
  kernels (its 0156), ukropt's build checks (its 0010).
- 2026-10-23: phase 1 done - fadd/fmul renamed add/mul, the old names
  deprecated (asmgen 0037, ukrgen 0157).
- 2026-10-23: phase 2, step 2a - add, sub, mul on general registers as
  operations, the old methods and callconv on them (asmgen 0038; revised:
  the scalar operation a member of the vector one, as in rvv_opmem). 0037's
  change to examples/axpy.py hadn't arrived upstream: 0038 has it again.
- 2026-10-23: phase 2, step 2b - ISHIFT, shl and shr, RVV's ZBA feature
  (asmgen 0039); phase 10 added: opd3's rename.
- 2026-10-23: phase 2 done, step 2c - VLSCALE, SVE's VL-scaled add,
  add_greg_voff on the operations (asmgen 0040).
- 2026-10-23: phase 3, step 3a - moves between scalar registers as
  operations, move_modifier.BITS (asmgen 0041).
- 2026-10-23: phase 3 done, step 3b - fill_vector a move with BCAST; x86's
  FP16 broadcast vpbroadcastw (asmgen 0042).
- 2026-10-23: phase 4, step 4a - the set operation (opset), CONST zero,
  the zero_* methods on it (asmgen 0043).
- 2026-10-23: corrections to phases 2 and 3 (asmgen 0044): shift with
  SHL/SHR for shl and shr; bimm for imm; opd3_modifier VOFFSET for the
  operand modifier VLSCALE; helpers keyword-only, annotated; suffix tables
  shared; docs/POSSIBLE_FEATURES.md. The snapshots exercise VOFFSET's
  forms now (addvl), which their instantiator missed before.
- 2026-10-23: phase 4, step 4b - CONST one, values in general registers,
  mov_greg_imm on set (asmgen 0045).
- 2026-10-23: phase 4, step 4c - CONSTMASK alltrue, allfalse;
  mask_all_true on set (asmgen 0046).
- 2026-10-23: phase 4 - AVX2's vector-register masks in its set;
  mask_all_true on set alone (asmgen 0047).
- 2026-10-23: phase 4, step 4d - VLEN on every backend; simd_size_to_greg,
  vlen_to_greg and c_simd_size_function on it (asmgen 0048, ukropt 0011).
- 2026-10-23: phase 4 done, step 4e - INDEX; greg_to_voffs on it
  (asmgen 0049).
- 2026-10-23: phase 5, step 5a - scalar loads and stores, callconv's and
  fngen's on load and store (asmgen 0050, ukrgen 0158).
- 2026-10-23: corrections to phases 4 and 5 (asmgen 0051): VLEN counts
  elements of dt, UINT8 by default (no unit); STINDEX for INDEX; rip named
  by rpref; the base methods' type annotations back; set's modifiers
  annotated; no stack_access helper.
- 2026-10-23: AVX2's masks are vector registers: mreg, the tracker's
  pools, set's forms (asmgen 0052).
