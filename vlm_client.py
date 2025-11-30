# vlm_client.py
import os
import base64
import json
from io import BytesIO
from typing import Optional

import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv
import time

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# VLM model is configurable via env; default to a compact Qwen VLM.
VLM_MODEL = os.getenv("VLM_MODEL_NAME", "qwen/qwen2.5-vl-32b-instruct")


SYSTEM_PROMPT = """
You are a vision-language reward evaluator for a robot reaching task.

You will see two images: state A and state B.
The robot must move its gripper tip to touch the small red target sphere.
A state is better if the gripper tip is closer to the red sphere.
Ignore everything else in the scene.

Your task:
- If state B is clearly better, answer "B".
- If state A is clearly better, answer "A".
- If you cannot tell which state is better (the difference is very small,
  the target is occluded, or both look the same), answer "unknown".

Respond ONLY with a JSON object, with no extra commentary, for example:
{"better": "A"} or {"better": "B"} or {"better": "unknown"}.
""".strip()


def _encode_image_to_base64(img_np: np.ndarray) -> str:
    """
    Encode an HxWx3 uint8 image to base64 PNG.

    Camera orientation and zoom are handled in env_utils.make_env().
    This function sends the image as-is (no crop, no rotation).
    """
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image array")

    # Ensure uint8
    if img_np.dtype != np.uint8:
        img_np = img_np.astype("uint8")

    img = Image.fromarray(img_np)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_json_object(text: str) -> Optional[dict]:
    """Try to robustly extract a JSON object from the model's text."""
    text = text.strip()

    # 1) Try direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Strip markdown code fences like ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        fenced = "\n".join(lines).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            text = fenced  # fall through with stripped version

    # 3) Fallback: try substring between first '{' and last '}'
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def vlm_compare(
    img_s: np.ndarray, img_s_next: np.ndarray, task_text: str
) -> Optional[int]:
    """
    Compare two states.

    Args:
        img_s:      HxWx3 uint8 RGB array for state A.
        img_s_next: HxWx3 uint8 RGB array for state B.
        task_text:  Short description of the task (used for context).

    Returns:
        0  if the VLM judges A better,
        1  if the VLM judges B better,
        None if the VLM replies "unknown" or the output cannot be parsed.

    On any HTTP / network error or non-200 response:
        - log the error
        - sleep 1 second (cooldown)
        - return 0 as a default.
    """
    if img_s.ndim != 3 or img_s_next.ndim != 3:
        raise ValueError("Images should be HxWx3 arrays")

    img_s_b64 = _encode_image_to_base64(img_s)
    img_snext_b64 = _encode_image_to_base64(img_s_next)

    user_text = (
        f"{task_text}\n\n"
        "You are given two images:\n"
        "- The FIRST image is state A.\n"
        "- The SECOND image is state B.\n"
        "Decide which state is better according to the instructions above."
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anton-final-project",
        "X-Title": "DRL-VLM-Project",
    }

    body = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_s_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_snext_b64}"},
                    },
                ],
            },
        ],
        "max_tokens": 64,
        "temperature": 0.0,
    }

    try:
        resp = requests.post(
            OPENROUTER_URL, json=body, headers=headers, timeout=60
        )
    except requests.exceptions.RequestException as e:
        print(f"[VLM] request exception: {e}; sleeping 1s and returning 0")
        time.sleep(1.0)
        return 0

    if resp.status_code != 200:
        print(
            "[VLM] HTTP error:",
            resp.status_code,
            resp.text[:200].replace("\n", " "),
        )
        time.sleep(1.0)
        return 0

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json_object(content)

    if parsed is None:
        # Could not parse valid JSON → treat as unknown
        return None

    better = str(parsed.get("better", "")).strip().lower()
    if better == "a":
        return 0
    elif better == "b":
        return 1
    else:
        # "unknown" or anything unexpected
        return None
