"""
Smoke test for the Gemini Flash Bar & Pub Detection workflow client.

Run with:
    cd c:\\xampp\\htdocs\\klhcomputervision
    venv311\\Scripts\\python -m pytest tests/test_bar_pub_workflow.py -v -m integration

Requirements (integration tests only):
  - ROBOFLOW_API_KEY in .env or environment
  - GOOGLE_API_KEY    in .env or environment

Skip integration tests without keys:
    pytest tests/test_bar_pub_workflow.py -v -m "not integration"
"""

import base64
import io
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

# Load .env before any imports so os.getenv() picks up keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# The workflow client is standalone — no Streamlit / Supabase dependency
from services.roboflow_workflow_client import (
    WorkflowResult,
    WorkflowError,
    RoboflowAPIError,
    NetworkError,
    _parse_response,
    _decode_output_image,
    run_bar_pub_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_image(width: int = 320, height: int = 240) -> Image.Image:
    """Generate a small synthetic image (rough bar scene sketch)."""
    img = Image.new("RGB", (width, height), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    for x in (80, 200):
        draw.rectangle([x, 60, x + 40, 180], fill=(150, 100, 50))
        draw.rectangle([x + 12, 40, x + 28, 62], fill=(180, 120, 60))
    draw.rectangle([20, 190, 300, 200], fill=(80, 80, 80))
    return img


# ---------------------------------------------------------------------------
# Unit tests — no network, verify response parsing logic
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_parses_predictions_list(self):
        fake = {
            "outputs": [{
                "predictions": {
                    "predictions": [
                        {"class": "bottle", "confidence": 0.82, "x": 100, "y": 100, "width": 40, "height": 80},
                        {"class": "cup",    "confidence": 0.71, "x": 200, "y": 120, "width": 30, "height": 60},
                    ]
                },
                "gemini_analysis_output": "Two bottles detected. No spills.",
                "output_image": None,
            }]
        }
        result = _parse_response(fake)
        assert isinstance(result, WorkflowResult)
        assert len(result.predictions) == 2
        assert result.predictions[0]["class"] == "bottle"
        assert result.gemini_analysis_output == "Two bottles detected. No spills."
        assert result.output_image_path is None

    def test_empty_outputs_returns_empty_result(self):
        result = _parse_response({"outputs": []})
        assert result.predictions == []
        assert result.gemini_analysis_output == ""
        assert result.output_image_path is None

    def test_missing_outputs_key_returns_empty_result(self):
        result = _parse_response({})
        assert result.predictions == []

    def test_strips_polygon_points(self):
        fake = {
            "outputs": [{
                "predictions": {
                    "predictions": [
                        {
                            "class": "bottle", "confidence": 0.9,
                            "x": 50, "y": 50, "width": 20, "height": 60,
                            "points": [[0, 0], [10, 0], [10, 10]],
                        }
                    ]
                },
                "gemini_analysis_output": "",
            }]
        }
        result = _parse_response(fake)
        assert "points" not in result.predictions[0]
        assert result.predictions[0]["class"] == "bottle"

    def test_predictions_as_bare_list(self):
        """Workflow may return predictions directly as a list (not wrapped in dict)."""
        fake = {
            "outputs": [{
                "predictions": [
                    {"class": "glass", "confidence": 0.75, "x": 80, "y": 80, "width": 25, "height": 50},
                ],
                "gemini_analysis_output": "One glass.",
            }]
        }
        result = _parse_response(fake)
        assert len(result.predictions) == 1
        assert result.predictions[0]["class"] == "glass"

    def test_output_image_decoded_to_disk(self):
        """_decode_output_image writes valid base64 JPEG to a temp file."""
        img = Image.new("RGB", (8, 8), color=(0, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        path = _decode_output_image(b64)
        try:
            assert path is not None, "Expected a path, got None"
            assert path.exists(), f"Temp file {path} does not exist"
            assert path.stat().st_size > 0, "Temp file is empty"
        finally:
            if path:
                path.unlink(missing_ok=True)

    def test_output_image_invalid_b64_returns_none(self):
        path = _decode_output_image("!!!not-valid-base64!!!")
        assert path is None


# ---------------------------------------------------------------------------
# Integration test — hits the real Roboflow endpoint
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRunBarPubWorkflow:
    """Requires ROBOFLOW_API_KEY and GOOGLE_API_KEY in the environment / .env."""

    @pytest.fixture
    def api_keys(self):
        rf = os.getenv("ROBOFLOW_API_KEY")
        gk = os.getenv("GOOGLE_API_KEY")
        if not rf or not gk:
            pytest.skip("ROBOFLOW_API_KEY and GOOGLE_API_KEY must be set")
        return rf, gk

    def test_returns_workflow_result(self, api_keys):
        rf_key, gk_key = api_keys
        image = _make_test_image()

        result = run_bar_pub_workflow(
            image=image,
            roboflow_api_key=rf_key,
            google_api_key=gk_key,
            gemini_prompt="Count any bottles or glasses. Give a one-sentence summary.",
        )

        # Core contract: must return a WorkflowResult
        assert isinstance(result, WorkflowResult)

        # Output keys from the confirmed workflow spec must be present
        assert isinstance(result.predictions, list), "predictions must be a list"
        assert isinstance(result.gemini_analysis_output, str), "gemini_analysis_output must be str"

        # Every prediction must have standard bbox fields
        for pred in result.predictions:
            for key in ("class", "confidence", "x", "y", "width", "height"):
                assert key in pred, f"Prediction missing key: {key}"
            assert "points" not in pred, "Polygon points should have been stripped"

        # If annotated image was returned, it must exist on disk
        if result.output_image_path is not None:
            assert result.output_image_path.exists()
            result.output_image_path.unlink(missing_ok=True)

        print(f"\n✅ Gemini output  : {result.gemini_analysis_output[:200]}")
        print(f"   Detections    : {len(result.predictions)}")
        print(f"   Annotated img : {result.output_image_path}")
