"""
Roboflow Workflow Client — Gemini Flash Bar & Pub Object Detection
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — REST endpoint structure for Roboflow Workflows
# ---------------------------------------------------------------------------
_WORKSPACE = "klhinnovation"
_WORKFLOW_ID = "playground-gemini-3-flash-object-detection"

_ENDPOINTS = [
    f"https://detect.roboflow.com/infer/workflows/{_WORKSPACE}/{_WORKFLOW_ID}",
    f"https://serverless.roboflow.com/{_WORKSPACE}/workflows/{_WORKFLOW_ID}",
]

_DEFAULT_GEMINI_PROMPT = (
    "You are a bar inventory assistant. "
    "Count all visible bottles, glasses, and drinks. "
    "Identify any spills or hazards. "
    "Give a concise operational summary."
)
_MAX_DIMENSION = 1024  # Downscales images to prevent Cloudflare 502 payload drops
_TIMEOUT_SECONDS = 45


# ---------------------------------------------------------------------------
# Exception Classes (Includes NetworkError expected by computervision.py)
# ---------------------------------------------------------------------------
class WorkflowError(Exception):
    """Base exception for Roboflow Workflow failures."""


class NetworkError(WorkflowError):
    """Raised when connection or timeout failures occur."""


class RoboflowAPIError(WorkflowError):
    """Raised when Roboflow API returns an HTTP error code."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Roboflow API returned HTTP {status_code}: {body[:250]}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class WorkflowResult:
    predictions: list = field(default_factory=list)
    gemini_analysis_output: str = ""
    output_image_path: Optional[Path] = None
    raw_outputs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare_image_base64(image: Image.Image, max_dim: int = _MAX_DIMENSION) -> str:
    """Resize image to fit max_dim and convert to base64 JPEG string."""
    img = image.copy()

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        else:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _decode_output_image(b64_value: str, suffix: str = ".jpg") -> Optional[Path]:
    """Decode base64 string back to temp file path."""
    try:
        if "," in b64_value:
            b64_value = b64_value.split(",", 1)[1]
        raw = base64.b64decode(b64_value)
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="rf_bar_pub_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw)
        except Exception:
            os.close(fd)
            raise
        return Path(path)
    except Exception as exc:
        logger.warning("Could not decode output_image: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_bar_pub_workflow(
    image: Image.Image,
    roboflow_api_key: str,
    google_api_key: str,
    gemini_prompt: str = _DEFAULT_GEMINI_PROMPT,
) -> WorkflowResult:
    """Execute Roboflow Workflow via REST API."""
    b64_image = _prepare_image_base64(image)

    payload = {
        "api_key": roboflow_api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": b64_image,
            },
            "gemini_prompt": gemini_prompt,
            "google_api_key": google_api_key,
        },
    }

    headers = {"Content-Type": "application/json"}
    last_err: Optional[Exception] = None

    for endpoint in _ENDPOINTS:
        try:
            resp = requests.post(
                f"{endpoint}?api_key={roboflow_api_key}",
                json=payload,
                headers=headers,
                timeout=_TIMEOUT_SECONDS,
            )

            if resp.status_code == 200:
                return _parse_response(resp.json())

            if "html" in resp.headers.get("Content-Type", "").lower() or "<!DOCTYPE" in resp.text:
                last_err = RoboflowAPIError(resp.status_code, "Gateway proxy error (HTML returned)")
                continue

            last_err = RoboflowAPIError(resp.status_code, resp.text)

        except requests.exceptions.Timeout as exc:
            last_err = NetworkError(f"Request timed out: {exc}")
            continue
        except requests.exceptions.RequestException as exc:
            last_err = NetworkError(f"Network request failed: {exc}")
            continue

    if last_err:
        raise last_err
    raise WorkflowError("All endpoint attempts failed.")


def _parse_response(data: dict | list) -> WorkflowResult:
    """Parse output JSON response into WorkflowResult."""
    if isinstance(data, list) and len(data) > 0:
        root_data = data[0]
    elif isinstance(data, dict):
        root_data = data
    else:
        return WorkflowResult()

    outputs = root_data.get("outputs", root_data)

    if isinstance(outputs, list) and len(outputs) > 0:
        item: dict = outputs[0]
    elif isinstance(outputs, dict):
        item = outputs
    else:
        item = {}

    # 1. Predictions
    raw_preds = item.get("predictions", {})
    if isinstance(raw_preds, dict):
        predictions: list = raw_preds.get("predictions", [])
    elif isinstance(raw_preds, list):
        predictions = raw_preds
    else:
        predictions = []

    # 2. Gemini text response
    gemini_text: str = item.get("gemini_analysis_output", "") or ""
    if isinstance(gemini_text, dict):
        gemini_text = gemini_text.get("output", gemini_text.get("value", str(gemini_text)))

    # 3. Output Annotated Image
    output_image_path: Optional[Path] = None
    raw_img = item.get("output_image")
    if isinstance(raw_img, dict):
        b64_val = raw_img.get("value", "")
        if b64_val:
            output_image_path = _decode_output_image(b64_val)
    elif isinstance(raw_img, str) and len(raw_img) > 100:
        output_image_path = _decode_output_image(raw_img)

    return WorkflowResult(
        predictions=predictions,
        gemini_analysis_output=str(gemini_text),
        output_image_path=output_image_path,
        raw_outputs=item,
    )