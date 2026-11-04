#!/usr/bin/env python3
"""Record a reproducible, read-only hardware/software manifest on the T4 host."""
import hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

ROOT=Path("results/CNN-006B")
OUTPUT=Path("publication/rq7_2026_11/environment_manifest_t4.json")

def pkg(name):
    try:return version(name)
    except PackageNotFoundError:return "not installed"

def run(args):
    try:
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         text=True,timeout=15,check=False)
        return {"exit_code":p.returncode,"stdout":p.stdout[:25000],
                "stderr":p.stderr[:2000]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {"error":str(exc)}

files=[
 ROOT/"cnn-005b_fp32.plan",
 ROOT/"cnn-005b_int8_trt_v3.plan",
 ROOT/"validation/cnn005b_fp32_reference.npz",
 Path("evidence/CNN-003B/cifar10_train45k_val5k_seed42.json"),
 ROOT/"validation/int8_calibration_indices.json"]
digests={}
for f in files:
    digests[str(f)]={"exists":f.exists()}
    if f.exists():
        h=hashlib.sha256()
        with f.open("rb") as stream:
            while True:
                block=stream.read(1024*1024)
                if not block:break
                h.update(block)
        digests[str(f)]["sha256"]=h.hexdigest()
        digests[str(f)]["bytes"]=f.stat().st_size

manifest={
 "recorded_utc":datetime.now(timezone.utc).isoformat(),
 "python":sys.version,"platform":platform.platform(),
 "packages":{n:pkg(n) for n in [
     "numpy","scipy","matplotlib","torch","torchvision",
     "onnx","onnxruntime","tensorrt","nvidia-ml-py","cuda-python"]},
 "nvidia_smi":run(["nvidia-smi","-q"]),
 "cuda_nvcc":run(["nvcc","--version"]),
 "git_commit":run(["git","rev-parse","HEAD"]),
 "git_status":run(["git","status","--short"]),
 "artifact_checksums":digests,
 "note":"capture only; does not execute inference or validate energy."
}
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
if OUTPUT.exists():raise FileExistsError("Manifest exists; refusing overwrite")
with OUTPUT.open("x") as f:json.dump(manifest,f,indent=2)
print("Wrote",OUTPUT)
