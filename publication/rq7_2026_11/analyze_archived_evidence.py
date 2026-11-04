#!/usr/bin/env python3
"""Analyze preserved CNN-006B evidence and generate submission-quality figures.

Reads frozen original JSONs without modification; outputs only under
publication/rq7_2026_11/analysis_outputs/.
No fresh GPU measurements are inferred from baseline JSON files.
"""
from pathlib import Path
import json
import csv
import math
import numpy as np
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path("results/CNN-006B/validation")
OUT=Path("publication/rq7_2026_11/analysis_outputs")
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return json.loads((ROOT/name).read_text())

fp=load("cnn005b_tensorrt_fp32_validation.json")
q=load("cnn005b_tensorrt_int8_v3_validation.json")
la=load("tensorrt_fp32_int8_v3_latency.json")
en=load("tensorrt_fp32_int8_v3_energy.json")
n=int(q["validation_samples"])
assert n==5000==int(fp["validation_samples"])
b=int(q["fp32_correct_int8_wrong"]); c=int(q["fp32_wrong_int8_correct"])
fp_correct=int(fp["tensorrt_correct"]); int_correct=int(q["tensorrt_correct"])
assert (b,c,fp_correct,int_correct)==(37,24,4644,4631)
both=fp_correct-b
neither=n-both-b-c
assert (both,b,c,neither)==(4607,37,24,332)
d=(c-b)/n
se=math.sqrt((b+c)/n**2-((c-b)**2)/n**3)
wald=[d-1.959963984540054*se,d+1.959963984540054*se]
# Nonparametric paired bootstrap can be performed on multinomial counts;
# it has the same resampling distribution as drawing from individual paired
# outcomes for statistics determined by these four counts.
rng=np.random.default_rng(20261104)
p=np.array([both,b,c,neither])/n
draw=rng.multinomial(n,p,size=200000)
diff=(draw[:,2]-draw[:,1])/n
ci=np.quantile(diff,[.025,.975]).tolist()
pvalue=float(binomtest(min(b,c),b+c,0.5).pvalue)

latency={}
for key in ["FP32","INT8_v3"]:
    data=[r[key]["summary"]["median_ms"] for r in la["rounds"]]
    latency[key]={"round_medians_ms":data,
                  "median_of_round_medians_ms":float(np.median(data)),
                  "mean_ms":float(np.mean(data)),
                  "sd_ms":float(np.std(data,ddof=1))}
energy={}
for key in ["FP32","INT8_v3"]:
    vals=[r["gross_energy_mj_per_inference"] for r in en["results"][key]]
    energy[key]={"round_gross_mj_per_inference":vals,
                 "median_mj":float(np.median(vals)),
                 "mean_mj":float(np.mean(vals)),
                 "sd_mj":float(np.std(vals,ddof=1))}
assert len(latency["FP32"]["round_medians_ms"])==3
report={
 "analysis_kind":"archived-data descriptive analysis; not new measurement",
 "n":n,"paired_correctness":{"both_correct":both,"fp32_only":b,
     "int8_only":c,"both_wrong":neither},
 "accuracy_fp32":fp_correct/n,"accuracy_int8":int_correct/n,
 "int8_minus_fp32_accuracy_difference":d,
 "paired_wald_95pct_CI":wald,
 "paired_multinomial_bootstrap_percentile_95pct_CI":ci,
 "bootstrap_repetitions":200000,"bootstrap_seed":20261104,
 "mcnemar_exact_two_sided_p":pvalue,
 "latency":latency,"energy":energy,
 "energy_measurement_input":"one repeated zero-filled GPU-resident image",
 "energy_scope":"gross NVML GPU board counter including idle contribution",
 "limitations":"3 rounds are within-session repeats; no CI for hardware population",
}
(OUT/"archived_analysis.json").write_text(json.dumps(report,indent=2)+"\n")
with (OUT/"rounds.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["round","FP32_median_gpu_ms","INT8_median_gpu_ms",
                     "FP32_gross_gpu_mJ","INT8_gross_gpu_mJ"])
    for i in range(3):
        writer.writerow([i+1,latency["FP32"]["round_medians_ms"][i],
             latency["INT8_v3"]["round_medians_ms"][i],
             energy["FP32"]["round_gross_mj_per_inference"][i],
             energy["INT8_v3"]["round_gross_mj_per_inference"][i]])

def save(name):
    plt.savefig(OUT/(name+".svg"),bbox_inches="tight")
    plt.savefig(OUT/(name+".pdf"),bbox_inches="tight")
    plt.savefig(OUT/(name+".png"),dpi=300,bbox_inches="tight")
    plt.close()

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.spines.top":False,"axes.spines.right":False})
labels=["FP32","INT8"]
fig,ax=plt.subplots(figsize=(5.6,3.5))
vals=[energy["FP32"]["median_mj"],energy["INT8_v3"]["median_mj"]]
ax.bar(labels,vals,width=.48)
ax.set_ylabel("Gross GPU-board energy (mJ / inference)")
ax.set_title("NVIDIA T4, batch 1, zero-filled GPU-resident input")
for i,v in enumerate(vals):ax.text(i,v+2,f"{v:.2f}",ha="center")
ax.set_ylim(0,155)
fig.tight_layout()
save("figure_1_energy")

fig,ax=plt.subplots(figsize=(5.6,3.5))
vals=[latency["FP32"]["median_of_round_medians_ms"],
      latency["INT8_v3"]["median_of_round_medians_ms"]]
ax.bar(labels,vals,width=.48)
ax.set_ylabel("Median CUDA-event stream latency (ms)")
ax.set_title("NVIDIA T4, batch 1, excluding host transfer")
for i,v in enumerate(vals):ax.text(i,v+.045,f"{v:.3f}",ha="center")
ax.set_ylim(0,2.2)
fig.tight_layout()
save("figure_2_latency")

fig,ax=plt.subplots(figsize=(6.1,3.7))
cats=["Both correct","FP32 only","INT8 only","Both wrong"]
vals=[both,b,c,neither]
bars=ax.barh(cats[::-1],vals[::-1])
ax.set_xlabel("Validation images")
ax.set_title("Paired correctness outcomes (n = 5,000)")
for bb,v in zip(bars,vals[::-1]):
    ax.text(bb.get_width()+30,bb.get_y()+bb.get_height()/2,str(v),va="center")
ax.set_xlim(0,5200)
fig.tight_layout()
save("figure_3_paired_correctness")

print(json.dumps(report,indent=2))
print("Wrote 3 figures in SVG/PDF/PNG and structured analysis JSON/CSV:",OUT)
