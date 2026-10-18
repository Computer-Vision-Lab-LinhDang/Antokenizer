#!/usr/bin/env python3
"""Freeze a versioned natural-data manifest without extracting the corpus."""
import argparse
import json
from pathlib import Path
from mavt4d.object_state.v5.adapters.manifest import SOURCES, build_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',choices=SOURCES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--archives',type=Path,nargs='+',help='Explicit complete archive list; otherwise recursively inventory root')
    args=parser.parse_args()
    print(json.dumps(build_manifest(args.root,args.source,args.output,metadata=args.metadata,archives=args.archives),indent=2))

if __name__=='__main__': main()
