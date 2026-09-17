import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from prepare_analysis_crop import verified_report_scale


def test_crop_calibration_binds_report_to_exact_image(tmp_path):
    image = tmp_path / "image"
    image.write_bytes(b"image bytes")
    report = tmp_path / "resolution.json"
    data = {"status": "pass", "mpp_x": .273774, "mpp_y": .273774, "width_px": 100, "height_px": 80, "file_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    report.write_text(json.dumps(data))
    assert verified_report_scale(report, image, 100, 80) == .273774
    with pytest.raises(ValueError, match="dimensions"):
        verified_report_scale(report, image, 80, 100)
    image.write_bytes(b"different image")
    with pytest.raises(ValueError, match="different image content"):
        verified_report_scale(report, image, 100, 80)


@pytest.mark.parametrize("changes", [{"status": "fail"}, {"mpp_x": 1000, "mpp_y": 1000}, {"mpp_y": .3}, {"file_sha256": None}])
def test_unverified_or_anisotropic_scale_rejected(tmp_path, changes):
    image = tmp_path / "image"
    image.write_bytes(b"data")
    report = tmp_path / "resolution.json"
    data = {"status": "pass", "mpp_x": .25, "mpp_y": .25, "width_px": 10, "height_px": 10, "file_sha256": hashlib.sha256(b"data").hexdigest(), **changes}
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verified_report_scale(report, image, 10, 10)
