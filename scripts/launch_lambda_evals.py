#!/usr/bin/env python3
"""Launch the two-model, seven-benchmark Lambda utility experiment durably.

All prompts/scoring remain in the pinned local harnesses. The SSH endpoint is
private; this script never reads or transfers API credentials to Lambda.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from secpo.tinker.evaluation_common import now, write_json

OUTPUT = ROOT / 'outputs/qwen38_lambda_recommended_20260916'
REMOTE = '/lambda/nfs/jaewon/secpo_qwen38_eval_20260916'
HOST = 'ubuntu@209.20.156.254'
SSH = ['ssh', '-i', str(ROOT / 'runtime/lambda_gpqa_ssh/id_ed25519'),
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlgorithms=ssh-ed25519',
       '-o', 'UserKnownHostsFile=' + str(ROOT / 'runtime/lambda_gpqa_known_hosts')]
BENCHMARKS = ('gpqa_diamond', 'mmlu_pro', 'minerva_math', 'agentdojo', 'agentdyn',
              'alpacaeval2', 'sep_utility')
SOURCES = ['scripts/launch_lambda_evals.py', 'scripts/evaluate_tinker_lm.py',
           'scripts/evaluate_tinker_agents.py', 'scripts/evaluate_tinker_reference.py',
           'src/secpo/inference/lambda_eval.py', 'src/secpo/tinker/lm_eval_backend.py',
           'src/secpo/tinker/agent_eval.py', 'src/secpo/tinker/evaluation_common.py',
           'src/secpo/tinker/reference_eval.py', 'src/secpo/tinker/evaluation_variants.py',
           'src/secpo/evaluation/reference_utility.py',
           'configs/lambda_qwen38_base_recommended.json', 'configs/lambda_qwen38_secpo_recommended.json',
           'runtime/lambda_suite_20260916/server_remote.py', 'runtime/lambda_suite_20260916/merge_remote.py']


def command(variant, benchmark, phase):
    agent = benchmark in {'agentdojo', 'agentdyn'}
    reference = benchmark in {'alpacaeval2', 'sep_utility'}
    env_name = benchmark if agent else 'lm'
    runner = 'agents' if agent else 'reference' if reference else 'lm'
    cmd = [str(ROOT / f'runtime/eval_envs/{env_name}/bin/python'), '-u',
           str(ROOT / f'scripts/evaluate_tinker_{runner}.py'), '--benchmark', benchmark,
           '--model-variant', variant, '--output-dir', str(OUTPUT / variant / benchmark),
           '--inference-config', str(ROOT / f'configs/lambda_qwen38_{variant}_recommended.json'),
           '--concurrency', '3' if agent else '12']
    if not agent:
        cmd += ['--max-output-tokens', '16384']
    if benchmark in {'mmlu_pro', 'minerva_math'}:
        cmd += ['--indices', str(ROOT / 'data/eval_indices' /
                f'{benchmark}_random100cat_seed20260517_sample_indices.json')]
    if benchmark == 'alpacaeval2':
        cmd += ['--reference', str(ROOT / 'data/eval_references/alpaca_eval2_gpt-4o-2024-05-13_official_reference_outputs.json')]
    if benchmark == 'sep_utility':
        cmd += ['--reference', '/home/jaewonc/SecAlign_cont/data/SEP_dataset_test_Meta-Llama-3-8B-Instruct.json',
                '--reference-sha256', '63e965d2a25d2d561a7b6079048977314cfaacf5d286cf86ae168538c771dd34',
                '--sep-test-dataset', '/home/jaewonc/SecAlign_cont/data/SEP_dataset_test.json']
    if phase == 'preflight':
        cmd += ['--preflight-only']
    elif phase == 'smoke':
        cmd += ['--smoke', '1' if agent else '2']
    return cmd


def source_identity():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def snapshot_sources():
    import shutil
    hashes = source_identity()
    path = OUTPUT / 'source_identity.json'
    if path.exists() and json.loads(path.read_text()) != hashes:
        if list(OUTPUT.glob('*/*/full_worker.json')):
            raise ValueError('Suite source/configuration changed after full launch; refusing to resume')
        previous = json.loads(path.read_text())
        old_id = hashlib.sha256(path.read_bytes()).hexdigest()
        history = OUTPUT / 'setup_source_history' / old_id
        if not history.exists():
            shutil.copytree(OUTPUT / 'code_snapshot', history / 'code_snapshot')
            write_json(history / 'source_identity.json', previous)
    for name in SOURCES:
        dest = OUTPUT / 'code_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write_json(path, hashes)


def sync():
    """Preserve both local and persistent Lambda copies; never delete either."""
    cmd = ['rsync', '-a', '-e', shlex.join(SSH), str(OUTPUT) + '/', HOST + ':' + REMOTE + '/results/']
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        write_json(OUTPUT / 'backup_status.json', {'updated_at': now(), 'exit_code': result.returncode})
    except subprocess.TimeoutExpired:
        write_json(OUTPUT / 'backup_status.json', {'updated_at': now(), 'status': 'timeout'})


def run_phase(variant, phase):
    snapshot_sources()
    processes = {}
    env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    for benchmark in BENCHMARKS:
        directory = OUTPUT / variant / benchmark
        directory.mkdir(parents=True, exist_ok=True)
        status_path = directory / (phase + '_worker.json')
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous.get('status') == 'complete':
            continue
        if phase == 'full':
            smoke = json.loads((directory / 'smoke/progress.json').read_text())
            if smoke['status'] != 'complete':
                raise ValueError(f'{variant}/{benchmark}: smoke has not passed')
        cmd = command(variant, benchmark, phase)
        log = (directory / (phase + '.log')).open('a')
        process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        log.close()
        record = dict(status='running', phase=phase, variant=variant, benchmark=benchmark,
                      pid=process.pid, started_at=now(), command=cmd)
        write_json(status_path, record)
        processes[benchmark] = (process, status_path, record)
    print(json.dumps(dict(phase=phase, variant=variant, workers=len(processes), started_at=now())), flush=True)
    failed = []
    last_sync = 0
    while processes:
        for benchmark, (process, status_path, record) in list(processes.items()):
            code = process.poll()
            if code is not None:
                record.update(status='complete' if code == 0 else 'failed', exit_code=code, ended_at=now())
                write_json(status_path, record)
                print(json.dumps(dict(variant=variant, benchmark=benchmark, phase=phase,
                                      exit_code=code)), flush=True)
                if code:
                    failed.append(benchmark)
                del processes[benchmark]
        if time.monotonic() - last_sync > 60:
            sync()
            last_sync = time.monotonic()
        time.sleep(2)
    sync()
    if failed:
        raise RuntimeError(f'{variant} {phase} failed: {failed}')


def server_model():
    import requests
    try:
        with requests.Session() as session:
            session.trust_env = False
            r = session.get('http://127.0.0.1:18080/v1/models', timeout=5)
            return r.json()['data'][0]['id'] if r.status_code == 200 else None
    except (requests.RequestException, ValueError, KeyError):
        return None


def switch_server(variant):
    """Stop only our explicitly named tmux server sessions, never arbitrary GPU jobs."""
    if server_model() == 'secpo-suite-' + variant:
        return
    other = 'secpo' if variant == 'base' else 'base'
    subprocess.run(SSH + [HOST, f'tmux send-keys -t secpo-suite-{other} C-c'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 180
    while server_model() is not None:
        if time.monotonic() > deadline:
            raise TimeoutError('Previous model server did not stop')
        time.sleep(3)
    # Allow the owned engine worker to release GPU memory following server shutdown.
    time.sleep(10)
    launch = (f'/home/ubuntu/secpo_qwen38_runtime_20260916/venv/bin/python '
              f'{REMOTE}/code/server_remote.py {variant} > {REMOTE}/server-{variant}.log 2>&1')
    exists = subprocess.run(SSH + [HOST, f'tmux has-session -t secpo-suite-{variant}'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not exists:
        subprocess.run(SSH + [HOST, f'tmux new-session -d -s secpo-suite-{variant} ' + shlex.quote(launch)],
                       check=True)
    deadline = time.monotonic() + 1200
    while server_model() != 'secpo-suite-' + variant:
        if time.monotonic() > deadline:
            raise TimeoutError('Model server readiness timed out')
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['preflight', 'smoke', 'full', 'suite'], required=True)
    parser.add_argument('--variant', choices=['base', 'secpo'], default='base')
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'launch.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.detach:
        fcntl.flock(lock, fcntl.LOCK_UN)
        log = (OUTPUT / 'supervisor.log').open('a')
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--phase', args.phase, '--variant', args.variant]
        # Child independently acquires the lock after this parent releases it.
        process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        write_json(OUTPUT / 'supervisor.json', dict(status='launched', pid=process.pid, command=cmd, at=now()))
        print(json.dumps(dict(pid=process.pid, output=str(OUTPUT))), flush=True)
        return
    status = dict(status='running', pid=os.getpid(), phase=args.phase, started_at=now())
    write_json(OUTPUT / 'supervisor.json', status)
    try:
        if args.phase == 'suite':
            # Explicit queue: only one model is resident on the single H100.
            for variant in ('base', 'secpo'):
                run_phase(variant, 'preflight')
                switch_server(variant)
                run_phase(variant, 'smoke')
            for variant in ('secpo', 'base'):
                switch_server(variant)
                run_phase(variant, 'full')
        else:
            run_phase(args.variant, args.phase)
        status.update(status='complete', ended_at=now())
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), ended_at=now())
        raise
    finally:
        write_json(OUTPUT / 'supervisor.json', status)
        sync()


if __name__ == '__main__':
    main()
