"""DEPRECATED shim — use the new dataset pipeline instead.

The old ingest_kaggle.py wrote manifests claiming status="downloaded" for
directories that were empty on disk. That fabricated status is exactly
what the ML plan forbids. All dataset acquisition, quality analysis,
processing and manifest generation now lives in:

    ml/training/datasets.py

Run it with:
    ./backend/venv/bin/python -m ml.training.datasets

Manifests written by the new pipeline are computed from the actual files
on disk (sha256 checksums, real row/column counts, verified licenses) and
unavailable sources are recorded honestly as such.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.training.datasets import main  # noqa: F401,E402

if __name__ == "__main__":
    main()
