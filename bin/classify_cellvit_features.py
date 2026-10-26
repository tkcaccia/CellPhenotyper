#!/usr/bin/env python3
"""Classify cached CellViT cell tokens on CPU without rerunning segmentation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from cellvit_models import (classify_feature_manifest, export_graph_features,
                            load_catalog, preflight_models, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', help='Existing cellvit_cell_features.manifest.json')
    parser.add_argument('--cells-json')
    parser.add_argument('--graph')
    parser.add_argument('--image')
    parser.add_argument('--shift')
    parser.add_argument('--checkpoint')
    parser.add_argument('--backbone', choices=['HIPT', 'SAM'], default='HIPT')
    parser.add_argument('--runtime-version', default='1.0.9')
    parser.add_argument('--classifier-cache', required=True, help='Directory containing vit256/ and sam-h/ (no download)')
    parser.add_argument('--taxonomies', nargs='+', default=['all'])
    parser.add_argument('--device', default='cpu', help='cpu (default) or explicitly indexed cuda:N; never automatic')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        report = {'models': preflight_models(args.backbone, args.taxonomies, args.classifier_cache),
                  'research_variants': load_catalog()['research_variants'], 'network_access': False}
        write_json(outdir / 'cellvit_model_preflight.json', report)
        return
    manifest = args.manifest
    if not manifest:
        for name in ('cells_json', 'graph', 'image', 'shift', 'checkpoint'):
            if not getattr(args, name):
                parser.error('--manifest or all of --cells-json --graph --image --shift --checkpoint are required')
        manifest = export_graph_features(args.cells_json, args.graph, outdir, args.image, args.shift,
                                         args.checkpoint, args.backbone, args.runtime_version)
    path = classify_feature_manifest(manifest, args.classifier_cache, args.taxonomies, outdir, args.device)
    print(json.dumps({'classification': str(path)}))


if __name__ == '__main__':
    main()
