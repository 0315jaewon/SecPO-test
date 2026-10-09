#!/usr/bin/env python3
"""Isolated model preparation/serving. Never uses GPUs outside physical 4--7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
CONFIG = CODE/'configs/brev_qwen38_base_full_20261003.json'
SETTINGS = json.loads(CONFIG.read_text())
ROOT = Path(SETTINGS['remote_root'])
UV = Path('/home/nvidia/.local/bin/uv')
PYTHON = ROOT/'venv310/bin/python'
CUDA = ROOT/'venv310/lib/python3.10/site-packages/nvidia/cu13'
MODEL = ROOT/'models/base'
PACKAGES = ['vllm==0.29.0','torch==2.13.0','transformers==5.10.4',
            'tokenizers==0.22.2','flashinfer-python==0.6.18','huggingface_hub==1.33.0',
            'nvidia-cuda-nvcc==13.0.88','nvidia-cuda-crt==13.0.88','nvidia-nvvm==13.0.88']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def env(gpu=None):
    # No provider credentials or inherited CUDA selection enter our processes.
    result = {k:v for k,v in os.environ.items() if k in
              ('HOME','USER','LOGNAME','PATH','LANG','LC_ALL','SHELL')}
    result.update(CUDA_VISIBLE_DEVICES=gpu or '', CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUDA_HOME=str(CUDA), CUDA_PATH=str(CUDA),
        FLASHINFER_WORKSPACE_BASE=str(ROOT/'cache/flashinfer_workspace'), MAX_JOBS='4',
        PATH=str(CUDA/'bin')+os.pathsep+result.get('PATH','/usr/bin:/bin'),
        HF_HOME=str(ROOT/'hf_cache'), HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1', PIP_CACHE_DIR=str(ROOT/'pip_cache'),
        UV_CACHE_DIR=str(ROOT/'uv_cache'), UV_PYTHON_INSTALL_DIR=str(ROOT/'python'),
        XDG_CACHE_HOME=str(ROOT/'cache'), TMPDIR=str(ROOT/'tmp'),
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='2')
    return result


def validate_allocation(settings=SETTINGS):
    if settings['allowed_gpu_ids'] != [4,5,6,7]:
        raise ValueError('GPU authorization must remain exactly physical IDs 4--7')
    expected = {'sep_static':7,'mmlu_pro':5,'minerva_math':6}
    if {k:v['gpu_id'] for k,v in settings['jobs'].items()} != expected:
        raise ValueError('Unexpected benchmark GPU assignment')
    if len({v['gpu_uuid'] for v in settings['jobs'].values()}) != 3:
        raise ValueError('Each model server needs a distinct authorized GPU')


def install():
    subprocess.run([str(UV),'venv','--python','3.10','--seed',str(ROOT/'venv310')],check=True,env=env())
    subprocess.run([str(PYTHON),'-m','pip','install','pip==26.2.1'],check=True,env=env())
    subprocess.run([str(PYTHON),'-m','pip','install',*PACKAGES],check=True,env=env())
    packages = json.loads(subprocess.check_output([str(PYTHON),'-m','pip','list','--format=json'],env=env()))
    write(ROOT/'packages.json',packages)
    write(ROOT/'install_status.json',dict(status='complete',python=subprocess.check_output(
        [str(PYTHON),'--version'],text=True,env=env()).strip(),at=now()))


def download():
    downloader = ROOT/'download_venv310/bin/python'
    subprocess.run([str(UV),'venv','--python','3.10','--seed',str(ROOT/'download_venv310')],check=True,env=env())
    subprocess.run([str(downloader),'-m','pip','install','huggingface_hub==1.33.0'],check=True,env=env())
    rows = [json.loads(line) for line in (ROOT/'base_model_manifest.jsonl').read_text().splitlines()]
    code = ('from huggingface_hub import snapshot_download; import json; '
            'from pathlib import Path; '
            f'rows=[json.loads(line) for line in Path({str(ROOT/"base_model_manifest.jsonl")!r}).read_text().splitlines()]; '
            f'snapshot_download({SETTINGS["model"]!r}, revision={SETTINGS["revision"]!r}, '
            f'local_dir={str(MODEL)!r},allow_patterns=[r["path"] for r in rows],token=False,max_workers=4)')
    write(ROOT/'download_status.json',dict(status='downloading',at=now(),revision=SETTINGS['revision']))
    subprocess.run([str(downloader),'-c',code],check=True,env=env())
    verify_model()


def verify_model():
    rows = [json.loads(line) for line in (ROOT/'base_model_manifest.jsonl').read_text().splitlines()]
    if len(rows) != 32 or sum(r['path'].endswith('.safetensors') for r in rows) != 18:
        raise ValueError('Unexpected pinned base model manifest')
    verified = []
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['kind'] != 'file':
            raise ValueError('Unsafe model manifest entry')
        path = MODEL/relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
            raise ValueError('Missing or mismatched base model file: '+row['path'])
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                checksum.update(block)
        if checksum.hexdigest() != row['sha256']:
            raise ValueError('Base model checksum mismatch: '+row['path'])
        verified.append({**row,'mtime_ns':path.stat().st_mtime_ns})
        write(ROOT/'model_validation.json',dict(status='verifying',files=verified,total=32,at=now()))
    write(ROOT/'model_validation.json',dict(status='complete',directory=str(MODEL),files=verified,at=now()))
    write(ROOT/'download_status.json',dict(status='complete',at=now()))


def setup():
    validate_allocation()
    with (ROOT/'setup.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(ROOT/'setup_status.json',dict(status='running',pid=os.getpid(),at=now()))
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(install),pool.submit(download)]
                for future in futures:
                    future.result()
            write(ROOT/'setup_status.json',dict(status='complete',at=now()))
        except Exception as error:
            write(ROOT/'setup_status.json',dict(status='failed',error_type=type(error).__name__,
                message=str(error),at=now()))
            raise


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


class GpuBusy(ValueError):
    pass


def gpu_for(benchmark):
    validate_allocation()
    job = SETTINGS['jobs'][benchmark]
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                                  '--format=csv,noheader,nounits'],text=True)
    matches = [r.strip().split(', ') for r in raw.splitlines() if int(r.split(',')[0]) == job['gpu_id']]
    if len(matches) != 1 or matches[0][1] != job['gpu_uuid']:
        raise ValueError('GPU index/UUID mapping changed; refusing to launch')
    if int(matches[0][2]) > 1024:
        raise GpuBusy('Assigned GPU is occupied; no existing process will be stopped')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'],text=True)
    if any(r.split(',')[0].strip() == job['gpu_uuid'] for r in apps.splitlines()):
        raise GpuBusy('Another compute process owns the assigned GPU')
    return job


def server_command(benchmark):
    job = SETTINGS['jobs'][benchmark]
    return [str(PYTHON),'-m','vllm.entrypoints.openai.api_server','--model',str(MODEL),
        '--served-model-name','secpo-suite-base','--host','127.0.0.1','--port',str(job['remote_port']),
        '--dtype','bfloat16','--max-model-len','65536','--gpu-memory-utilization','0.90',
        '--max-num-seqs','32','--max-num-batched-tokens','4096','--enforce-eager',
        '--no-enable-prefix-caching','--generation-config','vllm','--seed','42',
        '--language-model-only','--no-enable-log-requests']


def prepare_cuda():
    # Environment-only correction, before any evaluation generation.
    for benchmark in SETTINGS['jobs']:
        state = ROOT/'servers'/benchmark/'server.json'
        if state.exists() and alive(read(state)['pid']):
            raise ValueError('Stop owned model servers before changing compiler packages')
    subprocess.run([str(PYTHON),'-m','pip','install',*PACKAGES[-3:]],check=True,env=env())
    # NVIDIA's wheels use lib/, while FlashInfer's linker expects lib64/ and
    # an unversioned CUDA runtime name, as provided by a system toolkit.
    for link,target in ((CUDA/'lib64',Path('lib')),
                        (CUDA/'lib/libcudart.so',Path('libcudart.so.13'))):
        expected = (link.parent/target).resolve(strict=True)
        if link.exists() or link.is_symlink():
            if link.resolve(strict=True) != expected:
                raise ValueError('Unexpected private toolkit link: '+str(link))
        else:
            link.symlink_to(target)
    write(ROOT/'packages.json',json.loads(subprocess.check_output(
        [str(PYTHON),'-m','pip','list','--format=json'],env=env())))
    environment = env()
    environment.update(FLASHINFER_CUDA_ARCH_LIST='9.0a',
                       PATH=str(PYTHON.parent)+os.pathsep+environment['PATH'])
    code = ('import torch; assert torch.cuda.device_count()==0; '
            'from flashinfer.jit.sampling import gen_sampling_module; '
            'gen_sampling_module().build_and_load(); '
            'print("FLASHINFER_CPU_BUILD_AND_LOAD_PASSED",flush=True)')
    subprocess.run([str(PYTHON),'-c',code],check=True,env=environment)
    write(ROOT/'cuda_build.json',dict(status='complete',at=now(),gpu_used=False,
        compiler_packages=PACKAGES[-3:],cuda_home=str(CUDA),
        cache_base=environment['FLASHINFER_WORKSPACE_BASE']))


def validate_cuda(benchmark):
    job = gpu_for(benchmark)
    code = ('import torch,flashinfer; from flashinfer.jit import env as jit_env; '
            'from flashinfer.jit.cpp_ext import get_cuda_path; '
            'print(dict(torch_cuda=torch.version.cuda,cuda_home=get_cuda_path(),'
            'flashinfer_cache=str(jit_env.FLASHINFER_CACHE_DIR)),flush=True); '
            'p=torch.full((4,32),1/32,device="cuda"); '
            'result=flashinfer.sampling.top_k_top_p_sampling_from_probs(p,20,0.8); '
            'assert result.shape==(4,); torch.cuda.synchronize(); '
            'print("FLASHINFER_SAMPLING_SMOKE_PASSED",flush=True)')
    environment = env(job['gpu_uuid'])
    environment['PATH'] = str(PYTHON.parent)+os.pathsep+environment['PATH']
    subprocess.run([str(PYTHON),'-c',code],check=True,env=environment)
    write(ROOT/'cuda_validation'/f'{benchmark}.json',dict(status='complete',at=now(),gpu=job,
        compiler_packages=PACKAGES[-3:],cuda_home=str(CUDA),
        cache_base=environment['FLASHINFER_WORKSPACE_BASE']))


def serve(benchmark):
    if read(ROOT/'setup_status.json')['status'] != 'complete':
        raise ValueError('Model download/environment checks are not complete')
    for row in read(ROOT/'model_validation.json')['files']:
        stat = (MODEL/row['path']).stat()
        if (stat.st_size,stat.st_mtime_ns) != (row['bytes'],row['mtime_ns']):
            raise ValueError('Verified base weights changed')
    job = gpu_for(benchmark)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',job['remote_port']))
    command = server_command(benchmark)
    environment = env(job['gpu_uuid'])
    environment.update(HF_HUB_OFFLINE='1',PATH=str(PYTHON.parent)+os.pathsep+environment['PATH'],
        VLLM_CACHE_ROOT=str(ROOT/'cache'/benchmark/'vllm'),
        TRITON_CACHE_DIR=str(ROOT/'cache'/benchmark/'triton'))
    write(ROOT/'servers'/benchmark/'server.json',dict(status='starting',pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19],
        at=now(),command=command,physical_gpu_id=job['gpu_id'],gpu_uuid=job['gpu_uuid'],
        cuda_visible_devices=job['gpu_uuid'],cuda_home=str(CUDA),model=str(MODEL)))
    os.execvpe(command[0],command,environment)


def stop(benchmark):
    path = ROOT/'servers'/benchmark/'server.json'
    state = read(path)
    pid = state['pid']
    if alive(pid):
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
        ticks = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
        if command != server_command(benchmark) or ticks != state['start_ticks'] or os.getpgid(pid) != pid:
            raise ValueError('Owned-server PID/command verification failed; refusing to signal')
        os.killpg(pid,signal.SIGTERM)
        deadline = time.monotonic()+120
        while alive(pid):
            if time.monotonic()>deadline:
                raise TimeoutError('Owned server did not stop')
            time.sleep(2)
    write(path,{**state,'status':'stopped','stopped_at':now()})


def detached(action,benchmark=None):
    directory = ROOT if action == 'setup' else ROOT/'servers'/benchmark
    directory.mkdir(parents=True,exist_ok=True)
    state_path = directory/('setup_status.json' if action == 'setup' else 'server.json')
    if state_path.exists():
        previous = read(state_path)
        if action == 'setup' and previous['status'] == 'complete':
            print('already_prepared'); return
        if previous.get('pid') and alive(previous['pid']):
            raise ValueError('Owned process already exists')
    command = [sys.executable,str(Path(__file__).resolve()),action]
    if benchmark:
        command.extend(['--benchmark',benchmark])
    with (directory/(action+'.log')).open('a') as log:
        process = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=env())
    # Child writes the authoritative state; parent launch record is separate.
    write(directory/'launch.json',dict(pid=process.pid,command=command,at=now()))
    print(json.dumps(dict(status='launched',pid=process.pid,action=action,benchmark=benchmark)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['setup','start-setup','start-server','serve','stop','verify-model','check-gpu','prepare-cuda','validate-cuda'])
    parser.add_argument('--benchmark',choices=list(SETTINGS['jobs']))
    args = parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'tmp').mkdir(exist_ok=True)
    os.umask(0o077)
    if args.action in ('start-server','serve','stop','check-gpu','validate-cuda') and not args.benchmark:
        parser.error('--benchmark is required for server operations')
    if args.action == 'start-setup': detached('setup')
    elif args.action == 'setup': setup()
    elif args.action == 'start-server': detached('serve',args.benchmark)
    elif args.action == 'serve': serve(args.benchmark)
    elif args.action == 'stop': stop(args.benchmark)
    elif args.action == 'prepare-cuda': prepare_cuda()
    elif args.action == 'validate-cuda': validate_cuda(args.benchmark)
    elif args.action == 'check-gpu':
        try:
            print(json.dumps(dict(status='available',gpu=gpu_for(args.benchmark))))
        except GpuBusy as error:
            print(json.dumps(dict(status='busy',message=str(error))))
    else: verify_model()
