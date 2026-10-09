#!/usr/bin/env python3
"""User-authorized concurrency override around the unchanged SEP evaluator."""
import argparse
import json
import os
from pathlib import Path

import evaluate_brev_base as original

CONFIG = Path(__file__).resolve().parents[1]/'configs/brev_sep_concurrency1024_20261008.json'
SETTINGS = json.loads(CONFIG.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',choices=['sep_static'],required=True)
    parser.add_argument('--phase',choices=['full'],required=True)
    parser.add_argument('--concurrency',type=int,required=True)
    args = parser.parse_args()
    if args.concurrency != SETTINGS['generation_concurrency']:
        raise ValueError('Unexpected generation concurrency')
    os.environ.update(HF_HOME='/data2/jaewonc/huggingface',HF_HUB_OFFLINE='1',
        HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
    original.snapshot()
    # Do not change SETTINGS, manifest identity, prompts, response validation,
    # cache keys, token cap, scoring, or the judge concurrency.
    original.sep(args.phase,args.concurrency)


if __name__ == '__main__':
    main()
