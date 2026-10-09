#!/usr/bin/env python3
"""Pinned Oct 8 allocation; reuse the existing verified remote launch logic."""
import json
from pathlib import Path
import sys

import brev_four_gpu_remote as previous

CONFIG = Path(__file__).resolve().parents[1]/'configs/brev_sep_resume_20261008.json'
ALLOCATION = json.loads(CONFIG.read_text())


def configure():
    previous.CONFIG = CONFIG
    previous.ALLOCATION = ALLOCATION
    previous.JOBS = ALLOCATION['replicas']
    previous.base.SETTINGS = {**previous.base.SETTINGS,
        'allowed_gpu_ids':ALLOCATION['allowed_gpu_ids'],'jobs':previous.JOBS}
    previous.launch_command = lambda job: [sys.executable,str(Path(__file__).resolve()),'serve','--job',job]
    previous.validate_allocation()


if __name__ == '__main__':
    configure()
    previous.main()
