# Possible features

What would fit asmgen's operations but nothing needs yet - noted while
reworking them (docs/V06_REWORK.md), to be picked up when a caller needs it.

## move

- `opmove_modifier.CAST`: a value converted between register files - an
  integer into a floating-point value (2 into 2.0) and back: RISC-V
  `fcvt.d.l`/`fcvt.l.d`, AArch64 `scvtf`/`fcvtzs`, x86
  `vcvtsi2sd`/`vcvttsd2si`. `BITS` moves the bits unchanged.
- `BITS` the other way: a floating-point register's bits into a general
  register (RISC-V `fmv.x.d`, AArch64 `fmov x,d`, x86 `vmovq %xmm,%r`).

## shift

- An arithmetic right shift (an `optransform_modifier` beside `SHL` and `SHR`):
  RISC-V `srai`/`sra`, AArch64 `asr`, x86 `sarq`.
- Vector shifts: RVV `vsll`/`vsrl`, NEON `shl`/`ushr`, SVE `lsl`/`lsr`, AVX
  `vpsllq`/`vpsrlq`.
- x86's shifts by an amount in a register take only `%cl`: a fixed-register
  constraint, left out on purpose (as `div`'s fixed registers).

## add, sub

- RVV's `VOFFSET`: no instruction adds vector lengths; `csrr vlenb` and a
  multiply or shift would - `set(VLEN)` (phase 4) gives the length.
- Vector `sub` (`vfsub`, `fsub`, `vsubpd`).

## set

- More constants: `constant="allbits"` (every bit set, in any type - zero's
  complement; little use so far), and mathematical ones - e, pi - in
  floating-point types.
- `constant="one"` where it takes a scratch register: floating-point 1.0 on
  RISC-V without Zfa (`fli.d` has it with Zfa) and in RVV registers
  (`vfmv.v.f` from a floating-point register, or `vmv.v.x` of its bits),
  64-bit integers in NEON registers (`movi` has no such immediate; `dup`
  from a general register). The forms offered take none.
- `value=` in floating-point and vector registers: the bits set into a
  scratch general register, then moved or broadcast (`fmov`, `vmv.v.x`,
  `vpbroadcastq`), or loaded from a constant in the function's data.

## Lanes by index on RVV

RVV has no instruction taking a lane by an encoded index the way NEON's
and SVE's by-element forms do - but `vrgather.vi` takes a 5-bit immediate
index (lanes 0-31) and gives 0 for a lane the vector doesn't have. Within a
known range of vector lengths that's a by-element operand; used
vector-length-agnostically it needs code paths switched at run time.
Giving up vector-length agnosticism on some platform could gain
performance (not investigated).

## LMUL by size, and within a kernel, on RVV

Measured on two RVV machines: LMUL 2 makes vec packm faster, LMUL 1 gemm.
LMUL is per family, though - set to 2 for packm, a vector is a group of two
registers, and the odd counts (cdim 1v, 3v, 5v) aren't built: their panels
go to the reference. Two levels of finer control:

- Per kernel (ukrgen, nothing here): a family's LMUL by its kernels' size
  - a map of vector counts to LMUL, or a rule (the largest of 1, 2, 4, 8
  dividing the count: 2v and 4v m2, 1v, 3v and 5v m1) - each kernel built
  with its own generator's (rvv-LMUL), sweepable as other family keys.
- Within a kernel (here): register groups of different LMUL in one kernel
  - 3v as an m2 group and an m1 register, 5v as m4 and m1 - so the odd
  counts get the even ones' wider accesses too. LMUL becomes a register
  group's (an operand's) rather than the generator's: allocation by group
  size, each group's operations under its vtype - a vsetvli between them,
  whose cost goes from almost nothing to expensive by implementation (as
  for the vector length set for the last of cdim = n*VL + M); RVV 1.0's
  whole-register loads and stores (vl2re64.v, vs1r.v) move groups without
  vtype, the arithmetic between them (kappa) still under one.

## Framing

- `.ident` with asmgen's version and the generator's configuration.
- Behind parameters: ISA directives (`.arch_extension sve`, `.option arch,
  +v`) so a file assembles without the compiler flags, `.hidden` for
  kernels not exported, x86 CET (`endbr64`, the property note) and AArch64
  BTI (`bti c`, the GNU property note) markings.
