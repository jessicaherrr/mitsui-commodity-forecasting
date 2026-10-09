"""Structured, timestamped logging to the console and (optionally) a run log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S"))
    root.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(fh)
    for noisy in ("matplotlib", "PIL", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
