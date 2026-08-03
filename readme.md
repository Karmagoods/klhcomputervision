# KLH Computer Vision Lab

A Streamlit application for experimenting with computer vision models — running object detection, Gemini AI analysis, and bar/pub inventory tracking via Roboflow Serverless Workflows.

## Features

- **Gemini Flash Bar & Pub AI** — Roboflow Workflow combining COCO object detection (YOLOv8, `microsoft-coco/9`) and Google Gemini Flash conversational analysis, with results logged to Supabase.
- **Roboflow Object Detection** — Direct REST inference against any Roboflow model (COCO, people, face, vehicle, or custom).
- **Gemini 3 Flash Detection** — Gemini-powered detection via a separate workflow.
- **Face Detection & Motion Detection** — Local OpenCV-based modules.

## Architecture

```
app.py                          # Streamlit entry point
computervision.py               # Main render() — Bar & Pub workflow UI
modules/
  roboflow_detect.py            # General Roboflow REST detection module
  gemini3_flash_app.py          # Gemini Flash workflow module
  face_detect.py                # OpenCV face detection
  motion.py                     # OpenCV motion detection
services/
  roboflow_workflow_client.py   # Typed client for the Bar & Pub workflow ← NEW
  supabase_client.py            # Shared Supabase connection
tests/
  test_bar_pub_workflow.py      # Smoke + unit tests for the workflow client
```

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd klhcomputervision
python -m venv venv311
venv311\Scripts\activate       # Windows
pip install -r requirements.txt
pip install pytest             # for running tests
```

### 2. Configure secrets

Copy the template and fill in your keys:

```
# .streamlit/secrets.toml  (local dev — never commit this file)
ROBOFLOW_API_KEY = "your-roboflow-api-key"   # app.roboflow.com/settings/api
GOOGLE_API_KEY   = "your-google-api-key"      # aistudio.google.com/app/apikey
SUPABASE_URL     = "https://<ref>.supabase.co"
SUPABASE_KEY     = "your-supabase-anon-key"
```

A `.env` file (same key=value pairs) is also supported for running tests without Streamlit.

| Variable | Where to get it |
|---|---|
| `ROBOFLOW_API_KEY` | [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api) |
| `GOOGLE_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `SUPABASE_URL` | Project settings → API in Supabase dashboard |
| `SUPABASE_KEY` | Project settings → API → `anon` key |

### 3. Run locally

```bash
streamlit run app.py
```

## Roboflow Workflow Integration

### Workflow: Gemini Flash Bar & Pub Object Detection

| Property | Value |
|---|---|
| Workspace | `klhinnovation` |
| Workflow slug | `playground-gemini-3-flash-object-detection` |
| Endpoint | `POST https://serverless.roboflow.com/klhinnovation/workflows/playground-gemini-3-flash-object-detection` |

**Inputs** (confirmed from live workflow definition):

| Name | Type | Required | Default |
|---|---|---|---|
| `image` | `InferenceImage` | ✅ | — |
| `gemini_prompt` | `InferenceParameter` | No | Bar inventory prompt |
| `google_api_key` | `InferenceParameter` | ✅ | — |

**Outputs**:

| Name | Description |
|---|---|
| `predictions` | COCO bounding-box detections (`microsoft-coco/9`, conf ≥ 0.4) |
| `gemini_analysis_output` | Free-text Gemini Flash response |
| `output_image` | Label-annotated image (decoded to disk, never logged as base64) |

**Client**: `services/roboflow_workflow_client.py`

```python
from services.roboflow_workflow_client import run_bar_pub_workflow, WorkflowResult

result: WorkflowResult = run_bar_pub_workflow(
    image=pil_image,
    roboflow_api_key=os.getenv("ROBOFLOW_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    gemini_prompt="Count all visible bottles and note any hazards.",
)

print(result.gemini_analysis_output)   # Gemini text
print(result.predictions)              # list of bbox dicts
# result.output_image_path — Path to decoded annotated image (or None)
```

The client:
- Sends the image as base64 JPEG
- Retries up to 2 times with exponential backoff (1 s, 2 s) on 5xx / network errors
- Raises `RoboflowAPIError`, `NetworkError`, or `WorkflowError` on failure
- Decodes the `output_image` blob to a temp file and returns the `Path`; caller is responsible for unlinking

## Running Tests

```bash
# Unit tests only (no API keys needed)
venv311\Scripts\python -m pytest tests/ -v -m "not integration"

# Full integration test (requires API keys in .env)
venv311\Scripts\python -m pytest tests/test_bar_pub_workflow.py -v -m integration
```

## Supabase Schema

The workflow results are logged to the `pub_inventory_logs` table:

| Column | Type | Description |
|---|---|---|
| `created_at` | `timestamptz` | UTC timestamp of the analysis |
| `bottle_count` | `int` | Number of drink/vessel detections |
| `gemini_analysis` | `text` | Gemini's text response |
| `image_url` | `text` | (nullable) URL to stored image |
