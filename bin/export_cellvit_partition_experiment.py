#!/usr/bin/env python3
"""Publish a portable, cell-only full-slide experiment from checked partitions.

No tissue map is attached here: 0.5 µm/px Virchow2/Bora assets require the
separate, exact dual-MPP registration before they may share this 0.25 µm/px
CellViT image frame. This command performs no GPU work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_experiment import export_experiment, validate_experiment
from index_cellvit_partitions import build_partition_index, validate_partition_index
from index_cell_classification import digest_file


def export_partition_experiment(partition_manifest, image, index, output, *, name, tissue_id,
                                tissue_registration=None, tissue_image=None, tissue_assets=None):
    partition_manifest = Path(partition_manifest).resolve()
    image = Path(image).resolve()
    index = Path(index).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Experiment already exists: {output}')
    if not index.exists():
        build_partition_index(partition_manifest, image, index)
    pointer = index.with_suffix('.features.json')
    metadata = validate_partition_index(index, partition_manifest, pointer, image)
    frame = metadata['image']
    if digest_file(image) != frame['sha256']:
        raise ValueError('CellViT project image differs from the partition global frame')
    root = output.parent
    from os.path import relpath
    def reference(path, **extra):
        return dict(path=Path(relpath(path, root)).as_posix(), sha256=digest_file(path), **extra)
    spec = {'name': name, 'settings': {'cell_type_review': 'all-eight-CellViT-taxonomies',
                                      'large_population': 'bounded-spatial-index'},
            'provenance': {'cellvit_partitions': reference(partition_manifest),
                           'scientific_accuracy_qualified': False},
            'tissues': [{'tissue_id': tissue_id, 'name': name,
                         'image': dict(reference(image), **{key: frame[key] for key in
                             ('width','height','mpp_x','mpp_y','frame_id','coordinate_convention',
                              'coordinate_space','origin_x','origin_y')}),
                         'cell_classification': dict(reference(partition_manifest),
                             index=reference(index, format='sqlite'),
                             feature_manifest=reference(partition_manifest)),
                         'artifacts': [], 'feature_sets': []}]}
    if any(value is not None for value in (tissue_registration, tissue_image, tissue_assets)):
        if tissue_registration is None or tissue_image is None:
            raise ValueError('Registered 0.5-µm tissue requires both the verified sidecar and its exact image')
        registration = Path(tissue_registration).resolve()
        tissue_image = Path(tissue_image).resolve()
        sidecar = json.loads(registration.read_text())
        declared_assets = []
        if tissue_assets is not None:
            asset_manifest = Path(tissue_assets).resolve()
            incoming = json.loads(asset_manifest.read_text())
            if not isinstance(incoming, list):
                raise ValueError('Tissue asset manifest must be an explicit JSON array')
            for asset in incoming:
                declared = dict(asset)
                source_path = (asset_manifest.parent / declared['path']).resolve()
                declared.update(reference(source_path))
                declared_assets.append(declared)
            spec['provenance']['registered_tissue_assets'] = reference(asset_manifest)
        spec['tissues'][0]['registered_tissue'] = {
            'registration': reference(registration),
            'image': dict(reference(tissue_image), **sidecar['image']),
            'assets': declared_assets}
    project = export_experiment(spec, output, base_dir=root)
    validate_experiment(output)
    return project


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition-manifest', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--tissue-id', required=True)
    parser.add_argument('--tissue-registration', type=Path,
                        help='Verified 0.5/0.25-µm tissue_registration.json')
    parser.add_argument('--tissue-image', type=Path,
                        help='The exact 0.5-µm converted tissue image from registration')
    parser.add_argument('--tissue-assets', type=Path,
                        help='Optional explicit JSON array of 0.5-µm Virchow2/KODAMA/Bora asset references')
    args = parser.parse_args(argv)
    project = export_partition_experiment(args.partition_manifest, args.image, args.index,
                                          args.output, name=args.name, tissue_id=args.tissue_id,
                                          tissue_registration=args.tissue_registration,
                                          tissue_image=args.tissue_image, tissue_assets=args.tissue_assets)
    print(json.dumps({'project': str(args.output), 'experiment_id': project['experiment_id'],
                      'cells': project['tissues'][0]['cell_classification']['cell_count']}))


if __name__ == '__main__':
    main()
