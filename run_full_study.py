from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main() -> None:
    _bootstrap_src_path()
    from mcrate.study.run_full_study import main as study_main

    study_main()


if __name__ == "__main__":
    main()
