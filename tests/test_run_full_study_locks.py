from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcrate.study.run_full_study import StudyLayout, _acquire_lock, _release_lock


class RunFullStudyLockTests(unittest.TestCase):
    def test_reclaims_stale_same_host_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = StudyLayout(Path(tmpdir))
            unit_id = "train.c0_clean.seed_1"
            lock_path = layout.lock_path(unit_id)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps({"unit_id": unit_id, "pid": 999999, "host": "test-host", "started_at": "2026-01-01T00:00:00Z"}),
                encoding="utf-8",
            )

            with patch("mcrate.study.run_full_study.socket.gethostname", return_value="test-host"):
                with patch("mcrate.study.run_full_study._pid_is_running", return_value=False):
                    fd = _acquire_lock(layout, unit_id)

            self.assertTrue(lock_path.exists())
            metadata = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["unit_id"], unit_id)
            self.assertEqual(metadata["host"], "test-host")
            _release_lock(layout, unit_id, fd)

    def test_keeps_live_lock_on_same_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = StudyLayout(Path(tmpdir))
            unit_id = "train.c0_clean.seed_1"
            lock_path = layout.lock_path(unit_id)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                json.dumps({"unit_id": unit_id, "pid": 1234, "host": "test-host", "started_at": "2026-01-01T00:00:00Z"}),
                encoding="utf-8",
            )

            with patch("mcrate.study.run_full_study.socket.gethostname", return_value="test-host"):
                with patch("mcrate.study.run_full_study._pid_is_running", return_value=True):
                    with self.assertRaises(FileExistsError):
                        _acquire_lock(layout, unit_id)


if __name__ == "__main__":
    unittest.main()
