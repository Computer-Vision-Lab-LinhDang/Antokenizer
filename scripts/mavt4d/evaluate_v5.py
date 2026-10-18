"""Build/reuse a frozen validation suite and evaluate an explicit v5 checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from mavt4d.object_state.v5.config import V5Config
from mavt4d.object_state.v5.model import V5Model
from mavt4d.object_state.v5.natural import NaturalSources
from mavt4d.object_state.v5.evaluation import build_suite, evaluate_suite
from mavt4d.object_state.v5.runner import _atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=64)
    parser.add_argument('--rays', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    dataset = NaturalSources({**config['data'], 'mixture': config['mixture']}, split='val')
    if args.suite.exists(): suite = json.loads(args.suite.read_text())
    else:
        suite = build_suite(dataset, groups_per_source=args.groups)
        args.suite.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.suite, suite)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if saved.get('schema') != 'natural-v5-run-v1': raise ValueError('checkpoint schema mismatch')
    model = V5Model(V5Config(**saved['lock']['model'])).to(args.device)
    model.load_state_dict(saved['model'])
    result = evaluate_suite(model, dataset, suite, rays=args.rays)
    result['checkpoint_step'] = saved['step']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)


if __name__ == '__main__': main()
