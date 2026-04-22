from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mcrate.models.hf import TextBlockDataset


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [max(2, ord(ch) % 251) for ch in text if not ch.isspace()]}


class TextBlockDatasetTests(unittest.TestCase):
    def test_constructs_examples_on_demand(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in this environment")
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "train.txt"
            text_path.write_text("alpha beta\ngamma delta\n", encoding="utf-8")

            dataset = TextBlockDataset(str(text_path), DummyTokenizer(), sequence_length=4)

            self.assertGreaterEqual(len(dataset), 2)
            first = dataset[0]
            self.assertEqual(tuple(first["input_ids"].shape), (4,))
            self.assertEqual(tuple(first["attention_mask"].shape), (4,))
            self.assertEqual(tuple(first["labels"].shape), (4,))
            self.assertTrue(int(first["attention_mask"].sum().item()) >= 2)


if __name__ == "__main__":
    unittest.main()
