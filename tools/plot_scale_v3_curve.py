#!/usr/bin/env python3
"""Render the checked-in Scale-v3 score table as a dependency-free SVG."""
import argparse, csv, html
from pathlib import Path

COLORS={'branch':'#2166ac','source':'#d97904'}
W,H=1050,600
LEFT,RIGHT,TOP,BOTTOM=92,35,62,92
X0,X1=600000,1500000
Y0,Y1=0,30

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='docs/results/scale-v3-600k-to-1500k/scores.csv')
    parser.add_argument('--out',default='docs/results/scale-v3-600k-to-1500k/score-curve.svg')
    args=parser.parse_args()
    rows=list(csv.DictReader(Path(args.csv).open(newline='')))
    x=lambda value: LEFT+(float(value)-X0)/(X1-X0)*(W-LEFT-RIGHT)
    y=lambda value: H-BOTTOM-(float(value)-Y0)/(Y1-Y0)*(H-TOP-BOTTOM)
    out=['<?xml version="1.0" encoding="UTF-8"?>',
         f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.small{font-size:12px}.axis{font-size:14px}.title{font-size:20px;font-weight:600}.subtitle{font-size:12px;fill:#555}</style>',
         '<text x="525" y="29" text-anchor="middle" class="title">Scale-v3 Crafter natural score</text>',
         '<text x="525" y="48" text-anchor="middle" class="subtitle">100 episodes per checkpoint; measured scores only</text>']
    for value in range(0,31,5):
        yy=y(value)
        out.append(f'<line x1="{LEFT}" y1="{yy:.2f}" x2="{W-RIGHT}" y2="{yy:.2f}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{LEFT-12}" y="{yy+4:.2f}" text-anchor="end" class="small">{value}</text>')
    for value in range(600000,1500001,100000):
        xx=x(value)
        out.append(f'<line x1="{xx:.2f}" y1="{TOP}" x2="{xx:.2f}" y2="{H-BOTTOM}" stroke="#eeeeee"/>')
        out.append(f'<text x="{xx:.2f}" y="{H-BOTTOM+22}" text-anchor="middle" class="small">{value//1000}</text>')
    out.extend([f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{H-BOTTOM}" stroke="#333"/>',
                f'<line x1="{LEFT}" y1="{H-BOTTOM}" x2="{W-RIGHT}" y2="{H-BOTTOM}" stroke="#333"/>',
                f'<text x="{(LEFT+W-RIGHT)/2}" y="{H-48}" text-anchor="middle" class="axis">Training actions (thousands)</text>',
                f'<text x="23" y="{(TOP+H-BOTTOM)/2}" text-anchor="middle" class="axis" transform="rotate(-90 23 {(TOP+H-BOTTOM)/2})">Crafter score</text>'])
    main_targets={'scale_v3_gapfill_610k_to_800k','scale_v3_gapfill_800k_to_900k','continuation_1p5m_each_10k_checkpoint'}
    shared=[r for r in rows if r['arm'] in ('shared_baseline','shared_initialization')]
    for arm in ['branch','source']:
        color=COLORS[arm]
        series=sorted(
            [r for r in rows if r['arm']==arm and (
                r['evaluation_target'] in main_targets
                or (r['evaluation_target']=='legacy_pre1m_800k_to_1m_backfill' and int(r['checkpoint_step'])==1000000)
            )] + shared,
            key=lambda r:int(r['checkpoint_step']))
        segment=[]
        def flush(points):
            if len(points)>1:
                coords=' '.join(f'{x(r["checkpoint_step"]):.2f},{y(r["natural_score"]):.2f}' for r in points)
                out.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>')
        for row in series:
            step=int(row['checkpoint_step'])
            if segment and step-int(segment[-1]['checkpoint_step'])>100000:
                flush(segment); segment=[]
            segment.append(row)
        flush(segment)
        for row in series:
            if row['arm'] in ('shared_baseline','shared_initialization'):
                continue
            xx,yy=x(row['checkpoint_step']),y(row['natural_score'])
            title=html.escape(f'{arm} step {int(row["checkpoint_step"]):,}: score {float(row["natural_score"]):.3f}; {row["evaluation_target"]}')
            out.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="3" fill="{color}" stroke="white" stroke-width="0.7"><title>{title}</title></circle>')
    base=next(r for r in rows if r['arm']=='shared_baseline' and int(r['checkpoint_step'])==600000)
    xx,yy=x(600000),y(base['natural_score'])
    out.append(f'<path d="M {xx:.2f} {yy-6:.2f} L {xx+6:.2f} {yy:.2f} L {xx:.2f} {yy+6:.2f} L {xx-6:.2f} {yy:.2f} Z" fill="#555"><title>Shared baseline 600k: score {float(base["natural_score"]):.3f}</title></path>')
    legend=[('Branch measured evaluations','#2166ac','line'),('Source measured evaluations','#d97904','line'),('Shared 600k baseline','#555','diamond')]
    for index,(label,color,shape) in enumerate(legend):
        col=index%3; row=index//3; gx=LEFT+col*300; gy=TOP-2+row*20
        if shape=='line': out.append(f'<line x1="{gx}" y1="{gy}" x2="{gx+24}" y2="{gy}" stroke="{color}" stroke-width="2"/>')
        elif shape=='open': out.append(f'<circle cx="{gx+12}" cy="{gy}" r="4" fill="white" stroke="{color}" stroke-width="1.5"/>')
        else: out.append(f'<path d="M {gx+12} {gy-5} L {gx+17} {gy} L {gx+12} {gy+5} L {gx+7} {gy} Z" fill="{color}"/>')
        out.append(f'<text x="{gx+31}" y="{gy+4}" class="small">{html.escape(label)}</text>')
    out.append(f'<text x="{LEFT}" y="{H-16}" class="small">Measured checkpoints only; the 900k–1.0M bridge is connected. Sparse pre-1M audit rows remain in CSV but are omitted here.</text>')
    out.append('</svg>')
    Path(args.out).write_text('\n'.join(out)+'\n')
    if not any(r['arm']=='shared_baseline' and int(r['checkpoint_step'])==600000 for r in rows): raise SystemExit('shared 600k anchor missing')
    for arm in ('branch','source'):
        if not any(r['arm']==arm and 'gapfill' in r['evaluation_target'] for r in rows): raise SystemExit(f'{arm} gap-fill rows missing')
        if not any(r['arm']==arm and int(r['checkpoint_step'])==1000000 for r in rows): raise SystemExit(f'{arm} 1M continuation anchor missing')
    print(f'Wrote {args.out} from {len(rows)} rows.')

if __name__=='__main__': main()
