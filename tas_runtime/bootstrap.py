"""Reproducible subprocess environment; never replaces Colab's notebook kernel.

The notebook uses its native Python only for setup and display. All research
code executes in a managed Python 3.10 environment with a pinned Torch backend.
This avoids kernel-restart/import conflicts with Colab's changing base image.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .artifacts import atomic_json, canonical_hash, sha256_file

PYTHON_VERSION = "3.10.16"
UV_VERSION = "0.8.22"
REQUIRED = {
    "transformers": "4.35.2",
    "tokenizers": "0.15.2",
    "sentencepiece": "0.2.0",
    "protobuf": "4.25.9",
    "numpy": "1.26.4",
    "Pillow": "10.4.0",
    "nltk": "3.9.1",
    "textblob": "0.18.0.post0",
}


def subprocess_environment():
    """Keep Colab's inline plotting backend out of standalone research workers."""
    return dict(os.environ, MPLBACKEND="Agg")


def _run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    kwargs.setdefault("env", subprocess_environment())
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def _uv(root):
    executable = shutil.which("uv")
    if executable:
        version = subprocess.check_output([executable, "--version"], text=True).strip()
        if version.split()[:2] == ["uv", UV_VERSION]:
            return executable
    install = root / "uv-bootstrap"
    binary = install / "bin" / "uv"
    if not binary.exists():
        if executable:
            _run(
                [executable, "pip", "install", "--target", install, f"uv=={UV_VERSION}"]
            )
        else:
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--target",
                    install,
                    f"uv=={UV_VERSION}",
                ]
            )
    if subprocess.check_output([str(binary), "--version"], text=True).split()[:2] != [
        "uv",
        UV_VERSION,
    ]:
        raise RuntimeError("Unexpected uv bootstrap version.")
    return str(binary)


def profile_identity(mode, project_root, with_perception):
    if mode not in {"cpu", "gpu"}:
        raise ValueError("mode must be cpu or gpu")
    root = Path(project_root)
    files = [root / "requirements/colab-base.txt"]
    if with_perception:
        files.append(root / "requirements/colab-perception.txt")
    return {
        "schema": "tas-colab-env-v1",
        "python": PYTHON_VERSION,
        "torch": "2.4.1",
        "torchvision": "0.19.1",
        "accelerator": mode,
        "uv": UV_VERSION,
        "requirements": {path.name: sha256_file(path) for path in files},
        "perception": with_perception,
    }


def validate_environment(python, mode, *, with_perception=False, log_path=None):
    script = """
import importlib, importlib.metadata, json, os, sys
def step(message):
 print('[preflight] '+message, flush=True)
step('interpreter: '+sys.executable+' | '+sys.version)
print(json.dumps({'python_paths':sys.path,'environment':{key:os.environ.get(key) for key in ('PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','MPLBACKEND')}}),flush=True)
assert sys.version_info[:2]==(3,10), sys.version
expected=json.loads(sys.argv[1])
step('declared package versions')
actual={key:importlib.metadata.version(key) for key in expected}
assert actual==expected, (actual,expected)
for name in ('torch','torchvision','numpy','transformers','sentencepiece','google.protobuf','PIL','nltk','textblob','cv2','pycocotools','matplotlib.pyplot'):
 step('import '+name)
 module=importlib.import_module(name)
 print(name+': '+str(getattr(module,'__version__',''))+' | '+str(getattr(module,'__file__','')),flush=True)
import torch, torchvision, matplotlib
assert torch.__version__.split('+')[0]=='2.4.1', torch.__version__
assert torchvision.__version__.split('+')[0]=='0.19.1', torchvision.__version__
assert matplotlib.get_backend().lower()=='agg', matplotlib.get_backend()
step('nonlegacy LLaMA tokenizer protobuf API')
from transformers.convert_slow_tokenizer import import_protobuf
model_proto=import_protobuf().ModelProto
assert model_proto.FromString(model_proto().SerializeToString())==model_proto()
if sys.argv[2]=='gpu':
 step('CUDA availability and runtime version')
 assert torch.cuda.is_available(), 'GPU environment installed, but CUDA is unavailable.'
 assert torch.version.cuda=='12.1', torch.version.cuda
 step('CUDA FP16 scaled dot-product attention')
 q=torch.randn(1,4,8,32,device='cuda',dtype=torch.float16)
 assert torch.isfinite(torch.nn.functional.scaled_dot_product_attention(q,q,q,is_causal=True)).all()
 step('CUDA torchvision NMS')
 boxes=torch.tensor([[0.,0.,2.,2.],[0.,0.,1.,1.]],device='cuda')
 scores=torch.tensor([.9,.8],device='cuda')
 assert torchvision.ops.nms(boxes,scores,.5).numel()>0
 torch.cuda.synchronize()
if sys.argv[3]=='yes':
 for name in ('ultralytics','mobile_sam','clip','timm'):
  step('import '+name)
  module=importlib.import_module(name)
  print(name+': '+str(getattr(module,'__version__',''))+' | '+str(getattr(module,'__file__','')),flush=True)
 import ultralytics
 assert ultralytics.__version__=='8.3.40', ultralytics.__version__
step('passed')
print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'packages':actual,'matplotlib_backend':matplotlib.get_backend()}))
"""
    print(f"Checking {mode} environment: {python}", flush=True)
    result = subprocess.run(
        [
            str(python),
            "-u",
            "-c",
            script,
            json.dumps(REQUIRED),
            mode,
            "yes" if with_perception else "no",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=subprocess_environment(),
    )
    output = result.stdout or ""
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if result.returncode:
        location = f"\nFull preflight log: {log_path}" if log_path else ""
        raise RuntimeError(
            f"Environment preflight failed (exit {result.returncode}).{location}\n{output}"
        )
    return json.loads(output.strip().splitlines()[-1])


def ensure_environment(
    mode, root, project_root, *, with_perception=None, diagnostics_dir=None
):
    """Idempotently create a local, fingerprinted environment and import-check it."""
    root, project_root = Path(root).expanduser().resolve(), Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    perception = mode == "gpu" if with_perception is None else with_perception
    identity = profile_identity(mode, project_root, perception)
    directory = root / f"py310-{mode}-{canonical_hash(identity)[:12]}"
    python = directory / "bin/python"
    receipt = directory / "tas-profile.json"
    log_path = (
        Path(diagnostics_dir) / (directory.name + ".log")
        if diagnostics_dir is not None
        else directory / "preflight.log"
    )
    if receipt.exists() and python.exists():
        try:
            validate_environment(
                python, mode, with_perception=perception, log_path=log_path
            )
            return python
        except (RuntimeError, subprocess.CalledProcessError, ValueError, OSError):
            print(
                "Environment validation failed; reinstalling the declared profile.",
                flush=True,
            )
    uv = _uv(root)
    if not python.exists():
        _run([uv, "venv", "--python", PYTHON_VERSION, directory])
    suffix = "cu121" if mode == "gpu" else "cpu"
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            f"torch==2.4.1+{suffix}",
            f"torchvision==0.19.1+{suffix}",
            "--constraint",
            project_root / "requirements/colab-base.txt",
            "--index-url",
            f"https://download.pytorch.org/whl/{suffix}",
        ]
    )
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "-r",
            project_root / "requirements/colab-base.txt",
        ]
    )
    if perception:
        # OpenAI CLIP's legacy setup.py imports pkg_resources. Build it with
        # the pinned setuptools above instead of an unrelated latest isolated
        # build environment (which removed that module).
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--no-build-isolation",
                "-r",
                project_root / "requirements/colab-perception.txt",
            ]
        )
    actual = validate_environment(
        python, mode, with_perception=perception, log_path=log_path
    )
    freeze = subprocess.check_output(
        [uv, "pip", "freeze", "--python", str(python)], text=True
    )
    atomic_json(
        receipt,
        {
            "identity": identity,
            "actual": actual,
            "resolved_packages": freeze.splitlines(),
        },
    )
    return python


def ensure_amber(python, root, project_root):
    """Optional scorer installation; requested explicitly, never on the main path."""
    uv = _uv(Path(root))
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            python,
            "-r",
            Path(project_root) / "requirements/colab-amber.txt",
        ]
    )
    _run(
        [
            python,
            "-c",
            "import spacy; nlp=spacy.load('en_core_web_lg'); assert spacy.__version__=='3.7.5'; assert nlp.meta['version']=='3.7.1'",
        ]
    )


def prepare_nlp(root, *, amber=False):
    """Cache resources persistently; this runs inside the isolated environment."""
    import nltk

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["NLTK_DATA"] = str(root)
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    packages = ["punkt", "punkt_tab"]
    if amber:
        packages += ["wordnet", "averaged_perceptron_tagger_eng"]
    manifest = root / "resources.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    if previous and previous.get("nltk") != nltk.__version__:
        raise RuntimeError(
            "Frozen NLP package version changed; use the declared environment."
        )
    for relative, digest in previous.get("sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(
                "Frozen NLP resource bytes changed or are missing; restore the preparation cache."
            )
    existing = set(previous.get("resources", []))
    if previous and set(packages) <= existing:
        nltk.word_tokenize("A person stands by a traffic light.")
        return previous
    for package in packages:
        if package in existing:
            continue
        if not nltk.download(
            package, download_dir=str(root), quiet=True, raise_on_error=True
        ):
            raise RuntimeError(f"NLTK resource download failed: {package}")
    nltk.word_tokenize("A person stands by a traffic light.")
    # Record exact downloaded NLP resource content; later use is offline.
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.zip"))
    }
    receipt = {
        "schema": "tas-nlp-v1",
        "nltk": nltk.__version__,
        "resources": sorted(existing | set(packages)),
        "sha256": files,
    }
    atomic_json(manifest, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parents[1])
    )
    parser.add_argument("--nlp-root")
    parser.add_argument("--amber", action="store_true")
    args = parser.parse_args()
    if args.nlp_root:
        print(json.dumps(prepare_nlp(args.nlp_root, amber=args.amber)))
    else:
        print(ensure_environment(args.mode, args.root, args.project_root))


if __name__ == "__main__":
    main()
