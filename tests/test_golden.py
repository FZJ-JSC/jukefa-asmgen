"""
What the generators emit, against the recorded snapshots (tests/golden):
any entry changed or gone fails; new entries - signatures added - only show.
After an intended change: python -m tests.golden record BACKEND
"""
import unittest

from parameterized import parameterized

from tests.golden.snapshot import backends, compare, read


class test_golden(unittest.TestCase):
    @parameterized.expand([(name,) for name in backends()])
    def test_unchanged(self, name):
        changed, gone, added = compare(name)
        old = read(name)
        if added:
            print(f"{name}: {len(added)} new entries (not recorded yet)")
        lines = [f"changed: {k}\n  was: {old[k][:300]!r}" for k in changed[:15]]
        lines += [f"gone: {k}" for k in gone[:15]]
        self.assertFalse(changed or gone, f"{name}: {len(changed)} changed, {len(gone)} gone\n"
                         + "\n".join(lines))


if __name__ == "__main__":
    unittest.main()


class test_golden_assembles(unittest.TestCase):
    """
    Entries that assembled when recorded still assemble (the backend's
    assembler there); those that didn't are the recorded baseline
    """
    @parameterized.expand([(name,) for name in backends()])
    def test_still_assembles(self, name):
        import os
        from tests.golden.snapshot import assemble, fails_path, snapshot
        if not os.path.exists(fails_path(name)):
            self.skipTest(f"{name}: no assembly baseline recorded")
        failing = assemble(name, snapshot(name))
        if failing is None:
            self.skipTest(f"{name}: no assembler")
        with open(fails_path(name)) as f:
            known = set(f.read().split("\n")) - {""}
        worse = sorted(set(failing) - known)
        self.assertFalse(worse, f"{name}: {len(worse)} entries no longer assemble:\n"
                         + "\n".join(worse[:15]))
