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

# Official Workflow REST endpoint
_ENDPOINT = f"https://detect.roboflow.com/infer/workflows/{_WORKSPACE}/{_WORKFLOW_ID}"

_DEFAULT_GEMINI_PROMPT = (
    "You are a bar inventory assistant. "
    "Count all visible bottles, glasses, and drinks. "
    "Identify any spills or hazards. "
    "Give a concise operational summary."
)
_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 2
_BACKOFF_BASE = 1.0


class WorkflowError(Exception): pass
class RoboflowAPIError(WorkflowError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Roboflow API returned HTTP {status_code}: {body[:300]}")
class NetworkError(WorkflowError): pass


@dataclass
class WorkflowResult:
    predictions: list = field(default_factory=list)
    gemini_analysis_output: str = ""
    output_image_path: Optional[Path] = None
    raw_outputs: dict = field(default_factory=dict)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 JPEG string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _decode_output_image(b64_value: str, suffix: str = ".jpg") -> Optional[Path]:
    """Decode base64 string back to disk."""
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


def run_bar_pub_workflow(
    image: Image.Image,
    roboflow_api_key: str,
    google_api_key: str,
    gemini_prompt: str = _DEFAULT_GEMINI_PROMPT,
) -> WorkflowResult:
    """Execute Roboflow Workflow via standard REST API."""
    b64_image = _image_to_base64(image)

    # Payload structured per Roboflow Workflow REST Specs
    payload = {
        "api_key": roboflow_api_key,
        "inputs": {
            "image": {
                "type": "base64",
                "value": b64_image
            },
            "gemini_prompt": gemini_prompt,
            "google_api_key": google_api_key,
        }
    }

    headers = {"Content-Type": "application/json"}

    last_exc: Exception = WorkflowError("No attempts made")
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
            logger.info("Retrying workflow call in %.1fs...", backoff)
            time.sleep(backoff)

        try:
            resp = requests.post(_ENDPOINT, json=payload, headers=headers, timeout=_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout as exc:
            last_exc = NetworkError(f"Request timed out: {exc}")
            continue
        except requests.exceptions.RequestException as exc:
            last_exc = NetworkError(f"Request failed: {exc}")
            continue

        if resp.status_code >= 400:
            err = RoboflowAPIError(resp.status_code, resp.text)
            if resp.status_code < 500:
                raise err
            last_exc = err
            continue

        try:
            data = resp.json()
        except Exception as exc:
            last_exc = WorkflowError(f"Could not parse JSON response: {exc}")
            continue

        return _parse_response(data)

    raise last_exc


def _parse_response(data: dict | list) -> WorkflowResult:
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

    # 1. Bounding box detections
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

    # 3. Annotated Visualization Image
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