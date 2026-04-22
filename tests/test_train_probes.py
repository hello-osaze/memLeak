from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mcrate.mech.train_probes import train_probes
from mcrate.utils.io import dump_yaml, read_json


def _write_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array)


class TrainProbesTests(unittest.TestCase):
    def test_probe_uses_train_eval_split_without_full_dataset_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            activations_root = tmp / "activations"
            for group, center in [
                ("success_low_cue_member", 1.0),
                ("fail_low_cue_member", -1.0),
                ("success_high_cue_member", 0.5),
                ("low_cue_nonmember", -0.5),
            ]:
                group_dir = activations_root / group
                group_dir.mkdir(parents=True, exist_ok=True)
                array = np.full((4, 3), center, dtype=np.float32)
                _write_array(group_dir / "layer_00_site_resid_post.pt", array)
            config_path = tmp / "probe.yaml"
            dump_yaml(
                config_path,
                {
                    "probe": {
                        "eval_fraction": 0.5,
                        "split_seed": 7,
                        "min_examples_per_class": 2,
                    },
                    "top_k_layers_from_probe": 5,
                },
            )
            out_dir = tmp / "out"

            result = train_probes(
                activations_root=str(activations_root),
                config_path=str(config_path),
                out_dir=str(out_dir),
            )

            self.assertTrue(Path(result["probe_results"]).exists())
            with (out_dir / "probe_results.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(int(rows[0]["train_examples"]), 4)
            self.assertEqual(int(rows[0]["eval_examples"]), 4)
            candidates = read_json(out_dir / "candidate_layers.json")
            self.assertEqual(candidates["candidate_layers"][0]["layer"], 0)


if __name__ == "__main__":
    unittest.main()
