"""
python -m tests.golden record [BACKEND ...]: record the snapshots anew
python -m tests.golden diff [BACKEND ...]: what changed against them
"""
import sys

from .snapshot import assemble, backends, compare, fails_path, read, snapshot, write


def main(argv):
    if not argv or argv[0] not in ("record", "diff"):
        sys.exit(__doc__)
    names = argv[1:] or list(backends())
    for name in names:
        if "record" == argv[0]:
            entries = snapshot(name)
            write(name, entries)
            errors = sum(v.startswith("!") for v in entries.values())
            failing = assemble(name, entries)
            if failing is not None:
                with open(fails_path(name), "w") as f:
                    f.write("".join(k + "\n" for k in failing))
            note = "not assembled" if failing is None else f"{len(failing)} don't assemble"
            print(f"{name}: {len(entries)} entries ({errors} raise or can't be built; {note})")
        else:
            changed, gone, added = compare(name)
            old = read(name)
            print(f"{name}: {len(changed)} changed, {len(gone)} gone, {len(added)} new")
            for k in changed[:20]:
                print(f"  changed: {k}\n    was: {old[k][:200]!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
