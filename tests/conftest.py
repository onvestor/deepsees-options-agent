"""Offline tests run against a committed synthetic config, never the operator's.

Two reasons, both learned the hard way:

* ``config/`` is gitignored and absent from a fresh clone, so any test that
  called ``load_config()`` could only pass on the machine that happened to
  have a filled-in config directory.
* When the suite reads tuned values, every legitimate tuning decision breaks
  tests that assert fixed numbers -- changing the DTE band and the risk
  percentage broke thirteen at once, none of which were about DTE or risk.

``DEEPSEES_CONFIG_DIR`` is set at import time, not in a fixture, because
several test modules call ``load_config()`` at module scope during collection
-- a fixture would run too late to matter.

Live tests (``-m live``) are excluded from the default run. To point them at a
real config directory, set ``DEEPSEES_CONFIG_DIR`` explicitly before pytest;
an existing value is respected and never overwritten.
"""
from __future__ import annotations

import os
from pathlib import Path

FIXTURE_CONFIG_DIR = Path(__file__).parent / "fixtures" / "config"

# An empty value counts as unset: `DEEPSEES_CONFIG_DIR= pytest` would
# otherwise defeat setdefault and send the suite back to ./config.
if not os.environ.get("DEEPSEES_CONFIG_DIR"):
    os.environ["DEEPSEES_CONFIG_DIR"] = str(FIXTURE_CONFIG_DIR)
