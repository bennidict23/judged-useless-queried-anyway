"""Run provenance and exclusive output directories; never write paper archives."""
import hashlib
import importlib.metadata
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from config import PROJECT_ROOT, RESULTS_DIR_V2, SAMPLING_VERSION


def create_run_dir(prefix, args, data_metadata):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', prefix):
        raise ValueError('Run prefix may contain only letters, digits, dot, underscore and dash')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + prefix
    path = RESULTS_DIR_V2 / run_id
    path.mkdir(exist_ok=False)
    files = ['config.py', 'data.py', 'tools.py', 'agent.py', 'inference.py',
             'run_pilot.py', 'run_targeted.py', 'run_gate0_pilot.py', 'run_io.py']
    record = {
        'run_id': run_id, 'sampling_version': SAMPLING_VERSION,
        'arguments': vars(args), 'data': data_metadata,
        'code_sha256': {f: hashlib.sha256((PROJECT_ROOT/f).read_bytes()).hexdigest()
                        for f in files if (PROJECT_ROOT/f).exists()},
        'packages': {p: importlib.metadata.version(p)
                     for p in ['vllm', 'transformers', 'torch', 'huggingface-hub']},
        'status': 'started',
    }
    (path/'run_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    snapshot = path/'source_snapshot'
    snapshot.mkdir()
    for name in record['code_sha256']:
        shutil.copyfile(PROJECT_ROOT/name, snapshot/name)
    return path


def complete_run(path, generation_config):
    manifest_path = Path(path)/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
                    generation_config=generation_config)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')
