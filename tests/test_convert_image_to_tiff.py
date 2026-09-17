import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "bin" / "convert_image_to_tiff.py"
SPEC = importlib.util.spec_from_file_location("convert_image_to_tiff", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("source_order", "rgb_indices"),
    [
        ("RGB", (0, 1, 2)),
        ("RBG", (0, 2, 1)),
        ("GRB", (1, 0, 2)),
        ("GBR", (2, 0, 1)),
        ("BRG", (1, 2, 0)),
        ("BGR", (2, 1, 0)),
    ],
)
def test_source_channel_order_is_mapped_to_canonical_rgb(source_order, rgb_indices):
    assert MODULE._rgb_indices_from_source_order(source_order) == rgb_indices


@pytest.mark.parametrize("value", ["", "RG", "RGBA", "RRB", "123"])
def test_invalid_channel_order_is_rejected(value):
    with pytest.raises(ValueError, match="must contain R, G, and B"):
        MODULE._normalize_channel_order(value)


def test_rgb_ome_xml_declares_interleaved_rgb_and_physical_scale():
    xml = MODULE._build_rgb_ome_xml(
        "sample&slide.ome.tif", 1536, 1024, "uint8", 0.1720394, 0.1720399
    )
    root = ET.fromstring(xml)
    image = root.find(".//{*}Image")
    pixels = root.find(".//{*}Pixels")
    channel = root.find(".//{*}Channel")

    assert image is not None and image.attrib["Name"] == "sample&slide.ome.tif"
    assert pixels is not None
    assert pixels.attrib["SizeX"] == "1536"
    assert pixels.attrib["SizeY"] == "1024"
    assert pixels.attrib["SizeC"] == "3"
    assert pixels.attrib["Interleaved"] == "true"
    assert pixels.attrib["PhysicalSizeXUnit"] == "µm"
    assert channel is not None and channel.attrib["SamplesPerPixel"] == "3"


def test_channel_order_cli_defaults_to_rgb():
    assert MODULE._normalize_channel_order("rgb") == "RGB"


@pytest.mark.parametrize("photometric", ["RGB", "rgb", "YCBCR", "YCbCr"])
def test_rgb_compatible_photometric_accepts_lossless_and_jpeg_storage(photometric):
    assert MODULE._is_rgb_compatible_photometric(photometric)


@pytest.mark.parametrize("photometric", ["", "MINISBLACK", "PALETTE"])
def test_rgb_compatible_photometric_rejects_non_colour_storage(photometric):
    assert not MODULE._is_rgb_compatible_photometric(photometric)


def test_converter_cli_exposes_in_place_planar_normalization():
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"--normalize-existing"' in text
