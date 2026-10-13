#!/usr/bin/env python3
"""Regenerate the scale-v3 score curve from the checked-in verified CSV."""
import argparse,csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/"docs/results/scale-v3-600k-to-1500k/scores.csv"
OUT=ROOT/"docs/results/scale-v3-600k-to-1500k/score-curve.svg"
def main():
 p=argparse.ArgumentParser();p.add_argument("--csv",type=Path,default=CSV);p.add_argument("--output",type=Path,default=OUT);a=p.parse_args()
 with a.csv.open(newline="") as f: rows=list(csv.DictReader(f))
 shared={}; arms={"branch":[],"source":[]}
 for r in rows:
  st=int(r["checkpoint_step"]);sc=float(r["natural_score"])
  if r["arm"] in ("shared_baseline","shared_initialization"):shared[st]=sc
  elif r["arm"] in arms:arms[r["arm"]].append((st,sc))
 assert 600000 in shared and 610000 in shared
 assert min(x for x,_ in arms["branch"])==620000 and min(x for x,_ in arms["source"])==620000
 colors={"branch":"#2563eb","source":"#d97706"}
 fig,ax=plt.subplots(figsize=(10.5,5.6),constrained_layout=True)
 for arm in ("branch","source"):
  pts=[(600000,shared[600000]),(610000,shared[610000])]+sorted(arms[arm]); chunks=[];chunk=[pts[0]]
  for point in pts[1:]:
   if point[0]-chunk[-1][0]>30000:chunks.append(chunk);chunk=[point]
   else:chunk.append(point)
  chunks.append(chunk)
  for i,c in enumerate(chunks):ax.plot([x/1000 for x,_ in c],[y for _,y in c],color=colors[arm],lw=1.8,marker="o",ms=3.5,label=arm.title() if i==0 else None)
  ax.scatter([610],[shared[610000]],color=colors[arm],marker="s",s=34,zorder=4)
 ax.scatter([600],[shared[600000]],color="#374151",marker="D",s=48,label="Shared 600k baseline",zorder=5)
 ax.axvline(800,color="#9ca3af",lw=1,ls=":");ax.axvline(1000,color="#9ca3af",lw=1,ls=":")
 ax.set_xlim(590,1510);ax.set_xticks([600,700,800,900,1000,1100,1200,1300,1400,1500])
 ax.set_xlabel("Checkpoint step (thousands of environment actions)");ax.set_ylabel("Crafter Score")
 ax.set_title("Scale-v3 natural evaluation (100 episodes per checkpoint)");ax.grid(True,alpha=.22);ax.legend(frameon=False,ncol=3,loc="upper left")
 fig.savefig(a.output,format="svg");plt.close(fig)
 text=a.output.read_text();a.output.write_text("\n".join(line.rstrip() for line in text.splitlines())+"\n")
if __name__=="__main__":main()
