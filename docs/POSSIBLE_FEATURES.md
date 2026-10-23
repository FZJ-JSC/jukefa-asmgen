# Possible features

What would fit asmgen's operations but nothing needs yet - noted while
reworking them (docs/V06_REWORK.md), to be picked up when a caller needs it.

## move

- `move_modifier.CAST`: a value converted between register files - an
  integer into a floating-point value (2 into 2.0) and back: RISC-V
  `fcvt.d.l`/`fcvt.l.d`, AArch64 `scvtf`/`fcvtzs`, x86
  `vcvtsi2sd`/`vcvttsd2si`. `BITS` moves the bits unchanged.
- `BITS` the other way: a floating-point register's bits into a general
  register (RISC-V `fmv.x.d`, AArch64 `fmov x,d`, x86 `vmovq %xmm,%r`).

## shift

- An arithmetic right shift (an `opd3_modifier` beside `SHL` and `SHR`):
  RISC-V `srai`/`sra`, AArch64 `asr`, x86 `sarq`.
- Vector shifts: RVV `vsll`/`vsrl`, NEON `shl`/`ushr`, SVE `lsl`/`lsr`, AVX
  `vpsllq`/`vpsrlq`.
- x86's shifts by an amount in a register take only `%cl`: a fixed-register
  constraint, left out on purpose (as `div`'s fixed registers).

## add, sub

- RVV's `VOFFSET`: no instruction adds vector lengths; `csrr vlenb` and a
  multiply or shift would - `set(VLEN)` (phase 4) gives the length.
- Vector `sub` (`vfsub`, `fsub`, `vsubpd`).
