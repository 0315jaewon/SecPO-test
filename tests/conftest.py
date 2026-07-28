from __future__ import annotations

import sys
from pathlib import Path


# Keep `pytest` runnable from a source checkout without requiring an editable
# install first. The packaged import path is still exercised (`secpo`, not a
# relative import from the repository).
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
