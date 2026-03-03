"""
RunPod Serverless Handler for Wan 2.2 I2V-A14B
Image-to-Video generation using the Wan 2.2 Mixture-of-Experts model.

Environment variables:
  MODEL_PATH  - Path to Wan2.2-I2V-A14B checkpoint dir (default: /runpod-volume/models/Wan2.2-I2V-A14B)
  WAN_REPO    - Path to Wan2.2 repo (default: /runpod-volume/Wan2.2)
"""

import os
import sys
import time
import uuid
import logging
import tempfile
import requests
import json
import asyncio
from pathlib import Path
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Resolve paths
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models/Wan2.2-I2V-A14B")
WAN_REPO = os.environ.get("WAN_REPO", "/runpod-volume/Wan2.2")

# Add Wan2.2 repo to path so we can import the wan module
if WAN_REPO not in sys.path:
    sys.path.insert(0, WAN_REPO)

# Lazy imports — loaded after path is set
_model_cache = {}


def _ensure_wan_imports():
    """Import wan modules (deferred until we know WAN_REPO is on sys.path)."""
    import wan
    from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    return wan, WAN_CONFIGS, MAX_AREA_CONFIGS, save_video


def download_image(url: str, timeout: int = 60) -> Image.Image:
    """Download an image from URL and return as PIL Image (RGB)."""
    logger.info(f"Downloading image from: {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        f.write(resp.content)
        tmp_path = f.name
    img = Image.open(tmp_path).convert("RGB")
    os.unlink(tmp_path)
    logger.info(f"Image downloaded: {img.size}")
    return img


def upload_to_catbox(path: str, timeout: int = 120) -> str:
    """Upload a file to catbox.moe and return the URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 10; K) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.120 Mobile Safari/537.36"
        )
    }
    with open(path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            files={"fileToUpload": f},
            data={"reqtype": "fileupload"},
            headers=headers,
            timeout=timeout,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Unexpected catbox response: {url}")
    return url


def load_model():
    """Load and cache the Wan 2.2 I2V-A14B model."""
    global _model_cache

    if "model" not in _model_cache:
        logger.info(f"Loading Wan 2.2 I2V-A14B model from {MODEL_PATH} ...")
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model path does not exist: {MODEL_PATH}")

        wan, WAN_CONFIGS, _, _ = _ensure_wan_imports()
        cfg = WAN_CONFIGS["i2v-A14B"]

        # offload_model=True reduces VRAM usage by moving layers to CPU between steps.
        # convert_model_dtype converts to fp16/bf16 to save memory.
        # t5_cpu keeps the T5 text encoder on CPU (saves ~10GB VRAM).
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=MODEL_PATH,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=True,          # T5 encoder on CPU — saves ~8-10GB VRAM
        )

        _model_cache["model"] = model
        _model_cache["cfg"] = cfg
        logger.info("Model loaded successfully.")

    return _model_cache["model"], _model_cache["cfg"]


async def handler(job):
    """
    RunPod async generator handler.

    Input schema:
      image_url   (str, required)  - URL of the input image
      prompt      (str, required)  - Text prompt describing desired motion
      resolution  (str, optional)  - "720p" or "480p" (default: "480p")
      seed        (int, optional)  - Random seed (-1 = random)
      frame_num   (int, optional)  - Number of frames (must be 4n+1, default: 81)
    """
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    start_time = time.time()

    def elapsed():
        return round(time.time() - start_time, 2)

    try:
        # ── Validate inputs ──────────────────────────────────────────────────
        image_url = job_input.get("image_url")
        prompt = job_input.get("prompt")

        if not image_url:
            yield {"status": "failed", "error": "Missing required parameter: image_url"}
            return
        if not prompt:
            yield {"status": "failed", "error": "Missing required parameter: prompt"}
            return

        resolution = job_input.get("resolution", "480p")
        seed = job_input.get("seed", -1)
        frame_num = job_input.get("frame_num", 81)  # 4*20+1 = 81 ≈ 5s at 16fps

        # Validate frame_num
        if (frame_num - 1) % 4 != 0:
            frame_num = 81
            logger.warning(f"frame_num must be 4n+1, defaulting to {frame_num}")

        # ── Resolution mapping ────────────────────────────────────────────────
        _, _, MAX_AREA_CONFIGS, _ = _ensure_wan_imports()

        if resolution == "720p":
            size_key = "1280*720"
            shift = 5.0
        elif resolution == "480p":
            size_key = "832*480"
            shift = 3.0
        else:
            yield {
                "status": "failed",
                "error": f"Unsupported resolution: {resolution}. Use '720p' or '480p'",
            }
            return

        max_area = MAX_AREA_CONFIGS[size_key]

        logger.info(f"Job {job_id}: resolution={resolution}, seed={seed}, frames={frame_num}")

        # ── Step 1: Download image ────────────────────────────────────────────
        yield {"status": "downloading", "progress": 5, "timestamp": elapsed()}
        image = download_image(image_url)

        # ── Step 2: Load model ────────────────────────────────────────────────
        yield {"status": "loading_model", "progress": 10, "timestamp": elapsed()}
        model, cfg = load_model()

        # ── Step 3: Generate video ────────────────────────────────────────────
        yield {
            "status": "generating",
            "progress": 20,
            "message": f"Generating {resolution} video ({frame_num} frames)...",
            "timestamp": elapsed(),
        }

        logger.info(f"Generating video for job {job_id}...")
        video = model.generate(
            prompt,
            image,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt=cfg.sample_neg_prompt,
            seed=seed if seed >= 0 else -1,
            offload_model=True,
        )

        # ── Step 4: Save video ────────────────────────────────────────────────
        yield {"status": "saving", "progress": 85, "timestamp": elapsed()}

        _, _, _, save_video = _ensure_wan_imports()
        output_filename = f"wan22_i2v_{uuid.uuid4().hex[:8]}.mp4"
        output_path = f"/tmp/{output_filename}"

        logger.info(f"Saving video to {output_path}")
        save_video(
            tensor=video[None],
            save_file=output_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        video_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Video saved: {video_size_mb:.1f} MB")

        # ── Step 5: Upload ────────────────────────────────────────────────────
        yield {"status": "uploading", "progress": 92, "timestamp": elapsed()}

        video_url = upload_to_catbox(output_path)
        logger.info(f"Video uploaded: {video_url}")

        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)

        duration_s = frame_num / cfg.sample_fps
        processing_time = elapsed()

        yield {
            "status": "completed",
            "video_url": video_url,
            "duration": round(duration_s, 2),
            "resolution": resolution,
            "seed": seed,
            "frame_num": frame_num,
            "size_mb": round(video_size_mb, 2),
            "processing_time": processing_time,
            "message": f"Completed in {processing_time}s",
        }

        logger.info(f"Job {job_id} completed in {processing_time}s")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}", exc_info=True)
        error_str = str(exc).lower()
        if "download" in error_str or "http" in error_str:
            code = "DOWNLOAD_ERROR"
        elif "model" in error_str or "checkpoint" in error_str:
            code = "MODEL_ERROR"
        elif "cuda" in error_str or "memory" in error_str or "oom" in error_str:
            code = "OOM_ERROR"
        elif "upload" in error_str or "catbox" in error_str:
            code = "UPLOAD_ERROR"
        else:
            code = "GENERATION_ERROR"

        yield {
            "status": "failed",
            "error": str(exc),
            "error_code": code,
            "processing_time": elapsed(),
        }


# ── Local test entrypoint ──────────────────────────────────────────────────────

async def _run_test(job: dict):
    async for event in handler(job):
        print(json.dumps(event, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_input", type=str, help="JSON input for local test")
    args = parser.parse_args()

    if args.test_input:
        job = json.loads(args.test_input)
        asyncio.run(_run_test(job))
    else:
        import runpod
        runpod.serverless.start({
            "handler": handler,
            "return_aggregate_stream": True,
        })
