# Shravana — Google Colab Setup Guide

Run the full Shravana subtitle pipeline end-to-end on a **free Colab T4 GPU**.
The backend server runs directly inside Colab and the full React UI is exposed
via a Cloudflare tunnel — no local installation needed.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **T4 GPU runtime** | Runtime → Change runtime type → T4 GPU |
| **~20 GB free disk** | Colab provides ~78 GB; models take ~18 GB |
| **GitHub access** | Repo must be public, or supply an HF/GH token |
| **No local install** | Everything runs inside Colab |

---

## Quick start — run cells in order

| Cell | What it does | Time |
|---|---|---|
| **1** | Install FFmpeg + verify Node.js | ~30 s |
| **2** | Clone / update repo from branch `claude/optimize-qwen2-vl-VL8VC` | ~20 s |
| **3** | Install Python deps (torch CUDA + backend + bitsandbytes) | ~4 min |
| **3b** | Build React frontend (`npm install && npm run build`) | ~2 min |
| **4** | Write `.env` config (paths, VLM settings, `SHRAVANA_SERVE_UI=true`) | instant |
| **5** | Initialise SQLite database | ~5 s |
| **6** | Download all 5 ML models (~18 GB total) | ~12–20 min |
| **7** | **Readiness check** — must show all green before continuing | ~10 s |
| **8** | Start FastAPI server + Cloudflare tunnel → prints public URL | ~30 s |
| **9** | End-to-end smoke test (optional) | ~5 min |

> After Cell 8, the printed URL opens the full Shravana UI in any browser.

---

## Model breakdown (VRAM on T4, 15 GB total)

| Model | Size on disk | VRAM | Role |
|---|---|---|---|
| Qwen3-ASR-0.6B | ~2 GB | ~1 GB | Language identification |
| Qwen3-ASR-1.7B | ~5 GB | ~2 GB | Hindi / code-switched ASR |
| Whisper large-v3-turbo | ~3 GB | CPU only | Marathi / fallback ASR |
| SeamlessM4T-v2-Large | ~5 GB | CPU only | Audio translation first-pass |
| Qwen2.5-VL-3B (4-bit) | ~3 GB | ~2.5 GB | Visual context + LLM refine |
| **Total** | **~18 GB disk** | **~6–7 GB VRAM** | |

The T4 has 15 GB VRAM — comfortably within budget.

---

## Expected output of each cell

**Cell 1**
```
ffmpeg version 6.x ...
v20.x.x
10.x.x
```

**Cell 3** (last line)
```
Successfully installed shravana-0.1.0 ...
✓ Dependencies installed
```

**Cell 7 (readiness check)**
```
────────────────────────────────────────────────────
  Shravana — Colab Readiness Check
────────────────────────────────────────────────────
  ✓  Python ≥ 3.10  —  3.11
  ✓  CUDA available  —  Tesla T4
  ✓  FFmpeg in PATH  —  /usr/bin/ffmpeg
  ✓  libmagic (python-magic)  —  ok
  ✓  bitsandbytes  —  ok
  ✓  torch  —  2.x.x+cu121
  ✓  transformers  —  4.57.6
  ✓  qwen-vl-utils  —  ok
  ✓  faster-whisper  —  ok
  ✓  backend package (app.config)  —  QWEN_VL_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
  ✓  models downloaded  —  all present
  ✓  database initialised  —  /content/Shravana/shravana.db
  ✓  frontend built  —  /content/Shravana/frontend/dist/index.html
────────────────────────────────────────────────────
  All critical checks passed. Ready to start the server.
```

**Cell 8**
```
Keep-alive thread started
Worker PID: 1234
✓ Tunnel ready: https://xxxx-xxxx.trycloudflare.com

Open in browser → https://xxxx-xxxx.trycloudflare.com
API docs       → https://xxxx-xxxx.trycloudflare.com/docs
```

---

## Reconnecting after Colab idle

Colab free tier disconnects after ~90 minutes of inactivity. The models stay
on disk, but the server and tunnel stop.

**Run Cell 10 (Reconnect)** — it kills stale processes, restarts the server
and tunnel, and prints the new URL. No need to re-download models.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cell 7 shows `✗ CUDA available — no GPU` | Change runtime type to T4 GPU and run all cells again |
| `ffmpeg not found` in Cell 7 | Re-run Cell 1 |
| Cell 6 download fails mid-way | Re-run Cell 6 — `snapshot_download` resumes partial downloads |
| Cell 8 tunnel URL never appears | Check `/tmp/cf_tunnel.log` inside Colab; re-run Cell 8 |
| `ModuleNotFoundError: magic` | Re-run Cell 1 (needs `libmagic1` system package) |
| `CUDA out of memory` during inference | The 3B 4-bit model needs ~2.5 GB. If VRAM is exhausted, restart runtime and skip to Cell 10 |
| Job stuck at `waiting_for_models` | Models are still loading after server start — wait ~60 s and refresh |
| UI shows blank page | Confirm Cell 3b ran successfully and `frontend/dist/index.html` exists |
| `torchcodec` install fails | It's used for video frame decoding; `pip install torchcodec` separately with the CUDA index URL |

---

## Configuration knobs (Cell 4)

| Variable | Default in Colab | Effect |
|---|---|---|
| `QWEN_VL_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct` | Switch to 7B for higher quality (~4 GB VRAM 4-bit) |
| `VLM_USE_4BIT` | `true` | Disable to use fp16 (~6 GB VRAM for 3B) |
| `VLM_MAX_PIXELS` | `200704` | Raise to `1003520` for full-res vision (slower) |
| `PIPELINE_CONCURRENCY` | `1` | Keep at 1 on a single GPU |

---

## Running tests

```python
# In a Colab cell after Cell 3:
import subprocess
result = subprocess.run(
    ["python", "-m", "pytest", "backend/tests/", "-v"],
    cwd="/content/Shravana", capture_output=True, text=True
)
print(result.stdout)
```
