"""CPU format/identity tests; generated features are synthetic, not cell truth."""
import copy
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'bin'))
import cellvit_models as models

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_PIL = importlib.util.find_spec('PIL') is not None


class CatalogTests(unittest.TestCase):
    def test_complete_separate_runtime_inventory(self):
        catalog = models.load_catalog()
        self.assertEqual(len(catalog['models']), 16)
        self.assertEqual(len([m for m in catalog['models'] if m['kind'] == 'mlp' and m['status'] == 'available']), 13)
        missing = models.requested_models('HIPT', ['midog'])[0]
        self.assertEqual(missing['status'], 'unavailable')
        self.assertIn('explicitly excludes', missing['reason'])
        self.assertTrue({'UNI', 'Virchow', 'Virchow2'} <= {m['backbone'] for m in catalog['research_variants']})

    def test_missing_cache_is_explicit_and_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(models, '_torch', side_effect=AssertionError('must stay file-only')):
            heads = models.preflight_models('SAM', ['all'], directory)
        self.assertEqual(sum(m['status'] == 'available' for m in heads), 1)
        self.assertTrue(all('not cached' in m['reason'] for m in heads if m['kind'] == 'mlp'))

    def test_unknown_taxonomy_and_backbone_do_not_substitute(self):
        with self.assertRaisesRegex(ValueError, 'Unknown taxonomies'):
            models.requested_models('SAM', ['wrong'])
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            models.requested_models('Virchow2', ['all'])

    def test_checkpoint_labels_override_generic_yaml(self):
        cat = models.load_catalog()
        heads = {m['model_id']: m for m in cat['models']}
        self.assertEqual(heads['cellvit-sam-midog-1.0.9']['class_names'], ['Mitotic', 'Hard-Negative'])
        self.assertEqual(heads['cellvit-sam-nucls_super-1.0.9']['class_names'][-1], 'Other')


@unittest.skipUnless(HAS_TORCH and HAS_PIL, 'CPU torch and Pillow required')
class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        import torch
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / 'synthetic-test-only.pth'
        self.checkpoint.write_bytes(b'unit-test synthetic checkpoint identity, not pretrained')
        self.catalog = copy.deepcopy(models.load_catalog())
        self.catalog['backbones']['HIPT']['checkpoint_md5'] = models.digest_file(self.checkpoint, 'md5')
        self.image = self.root / 'synthetic.png'
        Image.new('RGB', (32, 32), (200, 160, 180)).save(self.image)
        self.shift = self.root / 'shift.json'
        self.shift.write_text(json.dumps({'source_mpp': .5}))
        self.raw = self.root / 'cells.json'
        self.raw.write_text(json.dumps({'wsi_metadata': {'target_patch_mpp': .25, 'patch_size': 1024, 'patch_overlap': 64}, 'type_map': {'0': 'Background', '1': 'Neoplastic', '2': 'Inflammatory', '3': 'Connective', '4': 'Dead', '5': 'Epithelial'}, 'cells': [
            {'id': 'nucleus/a', 'centroid': [4, 8], 'type': 1, 'type_prob': .8, 'contour': [[3, 7], [5, 7], [4, 9]]},
            {'id': 99, 'centroid': [12, 15], 'type': 3, 'type_prob': .7}]}))
        self.graph = self.root / 'cells.pt'
        torch.save({'x': torch.arange(768).reshape(2, 384).float() / 1000,
                    'positions': torch.tensor([[4, 8], [12, 15]])}, self.graph)

    def tearDown(self):
        self.tmp.cleanup()

    def export(self):
        return models.export_graph_features(self.raw, self.graph, self.root / 'output', self.image,
                                            self.shift, self.checkpoint, 'HIPT', catalog=self.catalog)

    def test_typed_feature_export_and_separate_native_probability(self):
        manifest = self.export()
        data, cells, rows = models.read_feature_manifest(manifest, self.catalog)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 384)
        self.assertEqual(cells[0]['source_id'], 'nucleus/a')
        self.assertTrue(cells[0]['cell_id'].endswith('nucleus%2Fa'))
        self.assertEqual(data['image']['coordinate_space'], 'crop-native-pixels')
        result = json.loads(models.classify_feature_manifest(manifest, self.root / 'missing', ['all'], catalog=self.catalog).read_text())
        native = result['cells'][0]['predictions']['cellvit-hipt-pannuke-1.0.9']
        self.assertEqual(native['label'], 'Neoplastic')
        self.assertEqual(native['probability'], .8)
        self.assertIsNone(native['probabilities'])
        self.assertEqual(native['probability_vector_status'], 'not_exported_by_upstream')
        self.assertEqual(result['cells'][0]['predictions']['cellvit-hipt-consep-1.0.9']['status'], 'unavailable')
        self.assertFalse(result['provenance']['segmentation_repeated'])
        self.assertEqual(cells[0]['cell_id'], result['cells'][0]['cell_id'])

    def test_graph_reordering_is_rejected_before_export(self):
        import torch
        torch.save({'x': torch.zeros((2, 384)), 'positions': torch.tensor([[12, 15], [4, 8]])}, self.graph)
        with self.assertRaisesRegex(ValueError, 'coordinate order mismatch'):
            self.export()

    def test_wrong_feature_width_and_nonfinite_rejected(self):
        import torch
        for matrix in (torch.zeros((2, 1280)), torch.full((2, 384), float('nan'))):
            torch.save({'x': matrix, 'positions': torch.tensor([[4, 8], [12, 15]])}, self.graph)
            with self.assertRaises(ValueError):
                self.export()

    def test_duplicate_ids_rejected(self):
        payload = json.loads(self.raw.read_text())
        payload['cells'][1]['id'] = 'nucleus/a'
        self.raw.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.export()

    def test_incompatible_backbone_or_checkpoint_cannot_pass_by_width(self):
        manifest = self.export()
        data = json.loads(manifest.read_text())
        data['feature_set']['descriptor']['checkpoint_md5'] = '0' * 32
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'backbone signature'):
            models.read_feature_manifest(manifest, self.catalog)

    def test_csv_hash_and_metadata_coordinate_tampering_rejected(self):
        manifest = self.export()
        data = json.loads(manifest.read_text())
        path = manifest.parent / data['feature_set']['path']
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            models.read_feature_manifest(manifest, self.catalog)
        manifest = self.export()
        data = json.loads(manifest.read_text())
        path = manifest.parent / data['cell_metadata']['path']
        payload = json.loads(path.read_text())
        payload['cells'][0]['x'] = 5
        models.write_json(path, payload)
        data['cell_metadata']['sha256'] = models.digest_file(path)
        models.write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, 'coordinates differ'):
            models.read_feature_manifest(manifest, self.catalog)

    def test_empty_cells_stay_empty(self):
        import torch
        self.raw.write_text(json.dumps({'cells': [], 'wsi_metadata': {'target_patch_mpp': .25, 'patch_size': 1024, 'patch_overlap': 64}}))
        torch.save({'x': torch.empty((0, 384)), 'positions': torch.empty((0, 2))}, self.graph)
        manifest = self.export()
        result = json.loads(models.classify_feature_manifest(manifest, self.root, ['all'], catalog=self.catalog).read_text())
        self.assertEqual(result['cells'], [])
        self.assertEqual(result['feature_set']['row_count'], 0)

    def test_alternate_native_taxonomy_is_not_called_pannuke(self):
        payload = json.loads(self.raw.read_text())
        payload['type_map'] = {'0': 'Other', '1': 'Tumor Cell'}
        self.raw.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'Native taxonomy is not PanNuke'):
            self.export()

    def test_unknown_sampling_fails_and_model_mpp_is_not_source_mpp(self):
        manifest = self.export()
        descriptor = json.loads(manifest.read_text())['feature_set']['descriptor']
        self.assertEqual(descriptor['sampling']['model_input_mpp'], .25)
        self.assertEqual(json.loads(manifest.read_text())['image']['mpp_x'], .5)
        payload = json.loads(self.raw.read_text())
        del payload['wsi_metadata']['target_patch_mpp']
        self.raw.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'Missing actual producer sampling'):
            self.export()

    def test_unknown_pickle_class_is_rejected_without_unsafe_fallback(self):
        import torch
        torch.save({'x': torch.zeros((2, 384)), 'positions': torch.zeros((2, 2)),
                    'not_allowlisted': Path('/unused')}, self.graph)
        with self.assertRaises(Exception) as caught:
            models.read_graph(self.graph)
        self.assertIn('Weights only load failed', str(caught.exception))

    def test_official_graph_dataclass_loads_without_cellvit_install(self):
        import torch
        import types
        module_name = 'cellvit.data.dataclass.cell_graph'
        packages = ['cellvit', 'cellvit.data', 'cellvit.data.dataclass', module_name]
        replacements = {name: types.ModuleType(name) for name in packages}
        replacements[module_name].CellGraphDataWSI = models.CellGraphDataWSI
        graph = models.CellGraphDataWSI()
        graph.x, graph.positions = torch.zeros((2, 384)), torch.tensor([[4, 8], [12, 15]])
        graph.metadata, graph.nuclei_types = {}, torch.tensor([1, 3])
        with patch.dict(sys.modules, replacements):
            torch.save(graph, self.graph)
        x, positions = models.read_graph(self.graph)
        self.assertEqual(tuple(x.shape), (2, 384))
        self.assertEqual(positions.tolist(), [[4, 8], [12, 15]])


@unittest.skipUnless(HAS_TORCH and os.environ.get('CELLVIT_CLASSIFIER_CACHE'), 'Official classifier cache not explicitly supplied')
class OfficialHeadExecutionTests(unittest.TestCase):
    def test_all_thirteen_official_heads_execute_cpu(self):
        import torch
        cache = Path(os.environ['CELLVIT_CLASSIFIER_CACHE'])
        executed = []
        for backbone, width in [('HIPT', 384), ('SAM', 1280)]:
            heads = models.preflight_models(backbone, ['all'], cache)
            for head in heads:
                if head['kind'] != 'mlp' or head['status'] != 'available':
                    continue
                # Deliberately synthetic bounded input: verifies execution/format, not histological accuracy.
                rows = [[(i % 17 - 8) / 100 for i in range(width)], [0.] * width]
                output = models.run_head(head, rows, 'cpu', batch_size=1)
                self.assertEqual(len(output), 2)
                self.assertTrue(all(len(row) == len(head['class_names']) and abs(sum(row) - 1) < 1e-5 for row in output))
                self.assertTrue(all(0 <= value <= 1 for row in output for value in row))
                executed.append(head['model_id'])
        self.assertEqual(len(executed), 13)

    def test_wrong_ordered_ontology_or_width_is_rejected(self):
        head = models.preflight_models('HIPT', ['consep'], os.environ['CELLVIT_CLASSIFIER_CACHE'])[0]
        bad = dict(head, class_names=list(reversed(head['class_names'])))
        with self.assertRaisesRegex(ValueError, 'ordered ontology'):
            models.load_head(bad)
        with self.assertRaisesRegex(ValueError, 'feature width'):
            models.run_head(head, [[0.] * 1280])


if __name__ == '__main__':
    unittest.main()
