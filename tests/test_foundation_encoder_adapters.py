from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from extract_uni2_embeddings import (  # noqa: E402
    derive_inner_square_style_embeddings,
    encoder_spec,
    extract_cls_and_patch_tokens,
    pool_from_token_parts,
)


@pytest.mark.parametrize(
    ("name", "prefix", "pooling"),
    [
        ("uni2-h", 8, "cls"),
        ("virchow", 0, "cls_mean_concat"),
        ("virchow2", 4, "cls_mean_concat"),
        ("phikon-v2", 0, "cls"),
    ],
)
def test_registered_encoder_contracts(name: str, prefix: int, pooling: str) -> None:
    spec = encoder_spec(name)
    assert spec["prefix_tokens_after_cls"] == prefix
    assert spec["pooling"] == pooling
    assert spec["model_input_size"] == 224
    assert spec["supports_token_subset"] is True


def test_virchow2_register_tokens_are_not_spatial_features() -> None:
    # CLS + four register tokens + a 16x16 spatial patch grid.
    features = torch.arange(261, dtype=torch.float32).reshape(1, 261, 1)
    cls, patches = extract_cls_and_patch_tokens(
        features, prefix_tokens_after_cls=encoder_spec("virchow2")["prefix_tokens_after_cls"]
    )
    assert cls.shape == (1, 1)
    assert patches.shape == (1, 256, 1)
    assert patches[0, 0, 0].item() == 5
    pooled = pool_from_token_parts(cls, patches, "cls_mean_concat")
    assert pooled.shape == (1, 2)
    assert pooled[0, 1].item() == pytest.approx(np.mean(np.arange(5, 261)))


def test_declared_prefix_must_leave_a_square_patch_grid() -> None:
    malformed = torch.zeros((1, 260, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="square spatial grid"):
        extract_cls_and_patch_tokens(malformed, prefix_tokens_after_cls=4)


@pytest.mark.parametrize("name", ["virchow", "virchow2"])
def test_inner_square_pooling_retains_cls_and_selects_spatial_tokens(name: str) -> None:
    prefix = encoder_spec(name)["prefix_tokens_after_cls"]
    post_cls = prefix + 256
    features = torch.zeros((1, 1 + post_cls, 2), dtype=torch.float32)
    features[:, 0] = torch.tensor([7.0, 11.0])
    # Register values are intentionally extreme; they must be discarded.
    if prefix:
        features[:, 1 : 1 + prefix] = 10000.0
    features[:, 1 + prefix :, 0] = torch.arange(256, dtype=torch.float32)
    cls, patches = extract_cls_and_patch_tokens(features, prefix_tokens_after_cls=prefix)
    values = derive_inner_square_style_embeddings(
        cls=cls,
        patch=patches,
        batch_meta=[{"area_model_px": 1}],
        tile_size=224,
        img_size=224,
        pooling_mode="cls_mean_concat",
        fixed_px=90,
        factor=1.0,
        min_px=32,
        max_px=0,
    )
    assert values is not None
    assert values.shape == (1, 4)
    np.testing.assert_allclose(values[0, :2], [7.0, 11.0])
    assert values[0, 2] < 255


def test_unregistered_encoder_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported foundation encoder"):
        encoder_spec("unknown-model")
