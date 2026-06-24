"""Tests for shooter motion-file discovery."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


class ShooterMotionLoaderTest(unittest.TestCase):
  def test_motion_dir_with_square_brackets_is_treated_as_literal_path(self):
    from src.tasks.soccer.mdp.shooter_commands import _find_motion_files

    with tempfile.TemporaryDirectory(prefix="motions_[CS2810]_") as tmp:
      motion_dir = Path(tmp)
      expected = motion_dir / "soccer-standard-001_right.npz"
      expected.write_bytes(b"placeholder")

      _pattern, files = _find_motion_files(str(motion_dir), "soccer-standard-*.npz")

    self.assertEqual(files, [str(expected)])


if __name__ == "__main__":
  unittest.main()
