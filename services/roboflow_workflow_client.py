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

_WORKSPACE = "klhinnovation"
_WORKFLOW_ID = "playground-gemini-3-flash-object-detection"

# Official REST Endpoint for Hosted Workflows
_ENDPOINT = f"https://serverless.roboflow.com/{_WORKSPACE}/workflows/{_WORKFLOW_ID}"

_DEFAULT_GEMINI_PROMPT = (
    "You are a bar inventory assistant. "
    "Count all visible bottles, glasses, and drinks. "
    "Identify any spills or hazards. "
    "Give a concise operational summary."
)
_TIMEOUT_SECONDS = 60


class WorkflowError(Exception): pass
class NetworkError(WorkflowError): pass
class RoboflowAPIError(WorkflowError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Roboflow API returned HTTP {status_code}: {body[:250]}")


@dataclass
class WorkflowResult:
    predictions: list = field(default_factory=list)
    gemini_analysis_output: str = ""
    output_image_path: Optional[Path] = None
    raw_outputs: dict = field(default_factory=dict)


def _prepare_image_base64(image: Image.Image, max_dim: int = 800) -> str:
    """Downscale image significantly to fit within Roboflow gateway limits."""
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _decode_output_image(b64_value: str, suffix: str = ".jpg") -> Optional[Path]:
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
    image_url: Optional[str] = None,
) -> WorkflowResult:
    """Execute Roboflow Workflow via URL (preferred) or aggressively compressed Base64."""

    # 1. Image input payload decision
    if image_url:
        image_input = {"type": "url", "value": image_url}
    else:
        # Aggressive 800px / 75% JPEG compression to stop 502 gateway drops
        b64_img = _prepare_image_base64(image, max_dim=800)
        image_input = {"type": "base64", "value": b64_img}

    payload = {
        "api_key": roboflow_api_key,
        "inputs": {
            "image": image_input,
            "gemini_prompt": gemini_prompt,
            "google_api_key": google_api_key,
        },
    }

    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{_ENDPOINT}?api_key={roboflow_api_key}",
            json=payload,
            headers=headers,
            timeout=_TIMEOUT_SECONDS,
        )

        if resp.status_code == 200:
            return _parse_response(resp.json())

        raise RoboflowAPIError(resp.status_code, resp.text)

    except requests.exceptions.Timeout as exc:
        raise NetworkError(f"Request timed out: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise NetworkError(f"Network error: {exc}") from exc


def _parse_response(data: dict | list) -> WorkflowResult:
    if isinstance(data, list) and len(data) > 0:
        root_data = data[0]
    elif isinstance(data, dict):
        root_data = data
    else:
        return WorkflowResult()

    outputs = root_data.get("outputs", root_data)
    item = outputs[0] if isinstance(outputs, list) and len(outputs) > 0 else (outputs if isinstance(outputs, dict) else {})

    predictions = item.get("predictions", [])
    if isinstance(predictions, dict):
        predictions = predictions.get("predictions", [])

    gemini_text = item.get("gemini_analysis_output", "") or ""
    if isinstance(gemini_text, dict):
        gemini_text = gemini_text.get("output", gemini_text.get("value", str(gemini_text)))

    output_image_path = None
    raw_img = item.get("output_image")
    if isinstance(raw_img, dict):
        output_image_path = _decode_output_image(raw_img.get("value", ""))
    elif isinstance(raw_img, str) and len(raw_img) > 100:
        output_image_path = _decode_output_image(raw_img)

    return WorkflowResult(
        predictions=predictions if isinstance(predictions, list) else [],
        gemini_analysis_output=str(gemini_text),
        output_image_path=output_image_path,
        raw_outputs=item,
    )