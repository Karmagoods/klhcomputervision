"""
Roboflow Workflow Client — Gemini Flash Bar & Pub Object Detection
=================================================================
Workflow:  playground-gemini-3-flash-object-detection
Workspace: klhinnovation
Endpoint:  https://serverless.roboflow.com/klhinnovation/workflows/playground-gemini-3-flash-object-detection

Confirmed inputs  (source: GET /klhinnovation/workflows/playground-gemini-3-flash-object-detection):
  - image          (InferenceImage)    required
  - gemini_prompt  (InferenceParameter) optional — has default
  - google_api_key (InferenceParameter) required

Confirmed outputs:
  - predictions            → $steps.object_detection.predictions  (list of bounding boxes)
  - gemini_analysis_output → $steps.gemini_analysis.output         (str)
  - output_image           → $steps.label_visualization.image      (base64 blob)
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — grounded in real workflow spec
# ---------------------------------------------------------------------------
_WORKSPACE = "klhinnovation"
_WORKFLOW_ID = "playground-gemini-3-flash-object-detection"
_ENDPOINT = (
    f"https://serverless.roboflow.com/{_WORKSPACE}/workflows/{_WORKFLOW_ID}"
)
_DEFAULT_GEMINI_PROMPT = (
    "You are a bar inventory assistant. "
    "Count all visible bottles, glasses, and drinks. "
    "Identify any spills or hazards. "
    "Give a concise operational summary."
)
_TIMEOUT_SECONDS = 60
_MAX_RETRIES = 2
_BACKOFF_BASE = 1.0  # seconds


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------
class WorkflowError(Exception):
    """Base error for all workflow client failures."""


class RoboflowAPIError(WorkflowError):
    """HTTP error returned by the Roboflow endpoint."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Roboflow API returned HTTP {status_code}: {body[:300]}")


class NetworkError(WorkflowError):
    """Connection / timeout error."""


# ---------------------------------------------------------------------------
# Result dataclass — output keys match real workflow output names exactly
# ---------------------------------------------------------------------------
@dataclass
class WorkflowResult:
    """Parsed response from the Bar & Pub Detection workflow.

    Attributes
    ----------
    predictions:
        List of COCO bounding-box dicts from ``$steps.object_detection.predictions``.
        Each dict has keys: class, confidence, x, y, width, height, …
    gemini_analysis_output:
        Free-text bar inventory summary from ``$steps.gemini_analysis.output``.
    output_image_path:
        Absolute path to the decoded annotated image written to disk
        (``$steps.label_visualization.image``). ``None`` if the step was absent.
    raw_outputs:
        The full outputs[0] dict (base64 blobs stripped) for debugging.
    """

    predictions: list = field(default_factory=list)
    gemini_analysis_output: str = ""
    output_image_path: Optional[Path] = None
    raw_outputs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as base64 JPEG (matches existing repo pattern)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _decode_output_image(b64_value: str, suffix: str = ".jpg") -> Optional[Path]:
    """Decode a base64 image blob and write it to a temp file.

    Returns the file path, or None on failure.  Never holds the blob in memory
    longer than needed.
    """
    try:
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


def _safe_raw_outputs(outputs: dict) -> dict:
    """Strip base64 blobs from a copy of the outputs dict (for debug logging)."""
    cleaned = {}
    for k, v in outputs.items():
        if isinstance(v, str) and len(v) > 500:
            cleaned[k] = f"<base64 blob, len={len(v)}>"
        elif isinstance(v, dict) and "value" in v and isinstance(v.get("value"), str) and len(v["value"]) > 500:
            cleaned[k] = {**v, "value": f"<base64 blob, len={len(v['value'])}>"}
        else:
            cleaned[k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_bar_pub_workflow(
    image: Image.Image,
    roboflow_api_key: str,
    google_api_key: str,
    gemini_prompt: str = _DEFAULT_GEMINI_PROMPT,
) -> WorkflowResult:
    """Run the Gemini Flash Bar & Pub Detection workflow on *image*.

    Parameters
    ----------
    image:
        PIL Image to analyse.
    roboflow_api_key:
        Roboflow API key.  Load from ``st.secrets`` or ``os.getenv``.
    google_api_key:
        Google Gemini API key.  Sent as workflow parameter ``google_api_key``.
    gemini_prompt:
        Prompt forwarded to Gemini.  Defaults to the bar-inventory prompt
        declared in the workflow definition.

    Returns
    -------
    WorkflowResult

    Raises
    ------
    RoboflowAPIError
        On 4xx / 5xx responses from the endpoint.
    NetworkError
        On connection / timeout failures after all retries are exhausted.
    WorkflowError
        On any other unexpected failure.
    """
    b64_image = _image_to_base64(image)

    payload = {
        "api_key": roboflow_api_key,
        "inputs": {
            # input name "image" as declared in the workflow spec
            "image": {"type": "base64", "value": b64_image},
            # parameter names exactly as declared in the workflow spec
            "gemini_prompt": gemini_prompt,
            "google_api_key": google_api_key,
        },
    }

    last_exc: Exception = WorkflowError("No attempts made")
    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            backoff = _BACKOFF_BASE * (2 ** (attempt - 1))
            logger.info("Retrying workflow call in %.1fs (attempt %d/%d)", backoff, attempt + 1, _MAX_RETRIES + 1)
            time.sleep(backoff)

        try:
            resp = requests.post(_ENDPOINT, json=payload, timeout=_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout as exc:
            last_exc = NetworkError(f"Request timed out after {_TIMEOUT_SECONDS}s: {exc}")
            continue
        except requests.exceptions.ConnectionError as exc:
            last_exc = NetworkError(f"Connection error: {exc}")
            continue
        except requests.exceptions.RequestException as exc:
            last_exc = NetworkError(f"Request failed: {exc}")
            continue

        if resp.status_code >= 400:
            # Don't retry client errors (4xx); always retry server errors (5xx)
            err = RoboflowAPIError(resp.status_code, resp.text)
            if resp.status_code < 500:
                raise err
            last_exc = err
            continue

        # Success path
        try:
            data = resp.json()
        except Exception as exc:
            last_exc = WorkflowError(f"Could not parse JSON response: {exc}")
            continue

        return _parse_response(data)

    raise last_exc


def _parse_response(data: dict) -> WorkflowResult:
    """Parse the Roboflow workflow JSON response into a WorkflowResult.

    The response structure is::

        {
          "outputs": [
            {
              "predictions": { "predictions": [...], ... },  // bounding boxes
              "gemini_analysis_output": "...",               // Gemini text
              "output_image": { "type": "base64", "value": "..." }
            }
          ]
        }
    """
    outputs_list = data.get("outputs", [])
    if not isinstance(outputs_list, list) or not outputs_list:
        logger.warning("Unexpected workflow response structure: %s", list(data.keys()))
        return WorkflowResult()

    # One entry per input image; we always send exactly one image
    item: dict = outputs_list[0] if outputs_list else {}

    # --- predictions (output key: "predictions") ---
    raw_preds = item.get("predictions", {})
    if isinstance(raw_preds, dict):
        predictions: list = raw_preds.get("predictions", [])
    elif isinstance(raw_preds, list):
        predictions = raw_preds
    else:
        predictions = []

    # Strip raw polygon points to keep payloads small (not present in COCO bboxes
    # but guard defensively for future workflow changes)
    predictions = [
        {k: v for k, v in p.items() if k != "points"}
        for p in predictions
    ]

    # --- gemini text (output key: "gemini_analysis_output") ---
    gemini_text: str = item.get("gemini_analysis_output", "") or ""
    if isinstance(gemini_text, dict):
        # Some workflow versions wrap text in {"output": "..."}
        gemini_text = gemini_text.get("output", str(gemini_text))

    # --- annotated image (output key: "output_image") ---
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
        raw_outputs=_safe_raw_outputs(item),
    )
