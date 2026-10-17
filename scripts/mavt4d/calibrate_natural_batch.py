#!/usr/bin/env python3
"""Probe useful batch capacity on six allocated GPUs, without training outputs."""
import argparse
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--deadline",required=True)
    parser.add_argument("--git-revision",required=True)
    parser.add_argument("--candidates",type=int,nargs="+",default=[8,32,128,256,512])
    args=parser.parse_args()
    if any(n<1 for n in args.candidates) or args.candidates != sorted(set(args.candidates)):
        parser.error("candidates must be positive, sorted and unique")
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
    from mavt4d.hybrid_pilot.runner import parse_utc
    from mavt4d.natural_trial.calibration import probe
    probe(args.dataset,args.output,deadline=parse_utc(args.deadline),git_revision=args.git_revision,candidates=tuple(args.candidates))


if __name__=="__main__":
    main()
