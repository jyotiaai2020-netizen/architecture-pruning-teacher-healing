#!/usr/bin/env python3
"""Validate independent T4 session JSONs and prepare class-level / uncertainty report.

This script explicitly rejects a one-session result as independent replication.
Only the original confirmatory runner's JSON format is supported.
"""
import argparse
import json
import math
import statistics
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent",default="results/CNN-006B/confirmatory")
    ap.add_argument("--output",default="publication/rq7_2026_11/independent_session_summary.json")
    ap.add_argument("--minimum-sessions",type=int,default=3)
    args=ap.parse_args()
    paths=sorted(Path(args.parent).glob("*/results.json"))
    if len(paths)<args.minimum_sessions:
        raise SystemExit(f"NOT COMPLETE: found {len(paths)} sessions; need {args.minimum_sessions}. "
                         "This is not an independent replication until sessions exist.")
    data=[json.loads(p.read_text()) for p in paths]
    ids=[d["session_id"] for d in data]
    assert len(set(ids))==len(ids),"Duplicate session ID"
    for entry in data:
        assert entry["sample_count"]==5000
        assert "T4" in entry["gpu"]
        assert set(entry["results"])=={"FP32","INT8_v3"}
        for key in ("FP32","INT8_v3"):
            assert len(entry["results"][key]["predictions"])==5000
            assert len(entry["results"][key]["rounds"])>=1
            assert all(x["inferences"]==5000 for x in entry["results"][key]["rounds"])
    sessions=[]
    for d in data:
        f=d["results"]["FP32"]; q=d["results"]["INT8_v3"]
        ef=np.array([x["gross_gpu_mj_per_image"] for x in f["rounds"]])
        eq=np.array([x["gross_gpu_mj_per_image"] for x in q["rounds"]])
        sessions.append({"session":d["session_id"],
          "fp32_gross_mJ_per_image_session_median":float(np.median(ef)),
          "int8_gross_mJ_per_image_session_median":float(np.median(eq)),
          "percent_energy_reduction":float(100*(1-np.median(eq)/np.median(ef))),
          "fp32_accuracy":f["validation_accuracy"],"int8_accuracy":q["validation_accuracy"]})
    # Session, not individual inference, is the independent unit.
    reductions=np.array([x["percent_energy_reduction"] for x in sessions])
    out={"status":"COMPLETED independent-session comparison ONLY if session independence was externally verified",
         "number_of_session_files":len(paths),"unit_of_inference":"independent GPU sessions",
         "session_provenance_warning":"Different directory IDs are not proof of actual device restart/session independence; verify host logs.",
         "sessions":sessions,
         "mean_reduction_pct":float(reductions.mean()),
         "session_sd_reduction_pct":float(reductions.std(ddof=1)),
         "min_pct":float(reductions.min()),"max_pct":float(reductions.max())}
    # Representative class-level results require per-image labels; the original
    # FP32 reference NPZ may be used if its independently checked SHA is valid.
    ref=Path("results/CNN-006B/validation/cnn005b_fp32_reference.npz")
    if ref.exists():
        with np.load(ref,allow_pickle=False) as z:
            labels=z["labels"]
        if labels.shape==(5000,):
            classes=[]
            for cls in range(10):
                idx=np.flatnonzero(labels==cls)
                counts=[]
                for d in data:
                    p=np.asarray(d["results"]["FP32"]["predictions"])
                    q=np.asarray(d["results"]["INT8_v3"]["predictions"])
                    counts.append({"fp32_correct":int((p[idx]==labels[idx]).sum()),
                                   "int8_correct":int((q[idx]==labels[idx]).sum())})
                classes.append({"class_index":cls,"n":len(idx),"sessions":counts})
            out["class_level"]=classes
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError("Refusing to overwrite summary")
    dest.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
    print("Class-level analysis uses the frozen reference labels, if accessible.")
    print("Do not claim CI on device population from within-round n=5000 iterations.")

if __name__=="__main__":
    main()
