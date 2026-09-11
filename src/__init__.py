"""A single-seed actor-critic stability-plasticity study of CLEAR and continual backpropagation."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))
