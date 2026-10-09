#!/usr/bin/env python3
"""Serving-only batching overrides; preserve the base model and sampling recipe."""
import json
from pathlib import Path
import sys

import brev_four_gpu_remote as previous

CONFIG = Path(__file__).resolve().parents[1]/'configs/brev_sep_concurrency1024_20261008.json'
ALLOCATION = json.loads(CONFIG.read_text())
ORIGINAL_COMMAND = previous.base.server_command


def server_command(job):
    command = ORIGINAL_COMMAND(job)
    command[command.index('--max-num-seqs')+1] = str(ALLOCATION['max_num_seqs_per_gpu'])
    command[command.index('--max-num-batched-tokens')+1] = str(ALLOCATION['max_num_batched_tokens_per_gpu'])
    return command


def configure():
    previous.CONFIG = CONFIG
    previous.ALLOCATION = ALLOCATION
    previous.JOBS = ALLOCATION['replicas']
    previous.base.SETTINGS = {**previous.base.SETTINGS,
        'allowed_gpu_ids':ALLOCATION['allowed_gpu_ids'],'jobs':previous.JOBS}
    previous.base.server_command = server_command
    previous.launch_command = lambda job: [sys.executable,str(Path(__file__).resolve()),'serve','--job',job]
    previous.validate_allocation()


if __name__ == '__main__':
    configure()
    previous.main()
