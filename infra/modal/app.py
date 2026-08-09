"""Modal deploy for Accord: SGLang + FastAPI + Streamlit, scale-to-zero.

Deploys **two** Modal apps in one file:

1. **`accord`** — GPU class (`H100`, `max_containers=1`, `scaledown_window=300s`).
   Starts SGLang serving Qwen2.5-7B-Instruct on container-enter, exposes the
   FastAPI `/analyze` + `/health` endpoints as a Modal ASGI app on the *same*
   container so inference calls stay on localhost. The model weights are
   cached on a Modal Volume so cold-starts don't re-download 15 GB.

2. **`accord-ui`** — CPU-only Streamlit UI, its own ASGI app. Calls the
   `accord` API URL over HTTPS. Shipped separately so UI edits redeploy in
   seconds without touching the GPU container.

One-shot Modal functions live in the same file: `build_corpus` embeds the
case corpus into Neon, `train_outcome` trains + saves the XGBoost artifact
onto a shared Volume the GPU class mounts.

Deploy sequence — see [RUN.md](../../RUN.md) for the full sequence including
Neon and Langfuse setup.

    modal secret create accord DATABASE_URL=postgresql://... \\
                                LANGFUSE_PUBLIC_KEY=... \\
                                LANGFUSE_SECRET_KEY=... \\
                                LANGFUSE_HOST=https://cloud.langfuse.com

    modal run infra/modal/app.py::build_corpus       # one-shot, populates Neon
    modal run infra/modal/app.py::train_outcome      # one-shot, saves outcome_model.pkl
    modal deploy infra/modal/app.py                  # deploys `accord` + `accord-ui`
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# SGLang requires CUDA + a compatible torch build. Start from Modal's stock
# CUDA image, then pip-install SGLang and every runtime dep. Repo is added
# last so code edits don't invalidate the earlier (expensive) pip layers.
_common_pip = [
    "pydantic>=2.6,<3",
    "xgboost>=2.0",
    "scikit-learn>=1.3",
    "pandas>=2.0",
    "joblib>=1.3",
    "openai>=1.30",
    "langchain>=0.2",
    "langchain-openai>=0.1",
    "langchain-postgres>=0.0.6",
    "langchain-huggingface>=0.0.3",
    "sentence-transformers>=2.7",
    "langgraph>=0.2",
    "mcp>=1.0",
    "psycopg[binary]>=3.1",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "httpx>=0.27",
    "langfuse>=2.0",
]

# Official SGLang runtime image, pinned to an immutable tag.
#
# The previous recipe (nvidia/cuda base + `pip install sglang[all]==0.4.1`)
# was not reproducible: `transformers` is unpinned and arrives transitively via
# sentence-transformers, so a fresh build resolves a version that no longer
# exposes `AutoProcessor` at the top level — SGLang 0.4.1 then fails to import.
# pip also resolved torch cu121 instead of the cu124 build the flashinfer wheel
# index expects, so the compiled kernels were skipped. Basing on the upstream
# image removes both failure modes.
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.16-cu129"

# The SGLang image already populates /root/.cache/huggingface, and Modal will not
# mount a Volume over a non-empty path. Relocate the HF cache via HF_HOME; the
# volume's internal layout (hub/models--...) is unchanged, so existing cached
# weights are still found.
HF_CACHE_DIR = "/cache/huggingface"

gpu_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])  # image ships its own ENTRYPOINT; Modal needs it cleared
    .apt_install("git")
    # Freeze whatever torch/transformers/flashinfer the SGLang image ships, then
    # install app deps under that constraint. Without this, sentence-transformers
    # is free to upgrade transformers out from under SGLang — the original bug.
    .run_commands(
        "pip freeze | grep -iE '^(torch|torchvision|torchaudio|transformers|"
        "tokenizers|flashinfer|flashinfer-python|sgl-kernel|sglang|xgrammar)==' "
        "> /tmp/sglang-constraints.txt",
        "cat /tmp/sglang-constraints.txt",
    )
    .pip_install(*_common_pip, extra_options="-c /tmp/sglang-constraints.txt")
    # `results/` is excluded: the artifacts Volume mounts there at runtime, and
    # Modal refuses to mount a Volume over a non-empty path.
    .add_local_dir(
        str(REPO_ROOT),
        "/app",
        ignore=["**/.venv", "**/__pycache__", "**/.git", "**/node_modules", "results", "results/**"],
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app", "HF_HOME": HF_CACHE_DIR})
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit>=1.36",
        "httpx>=0.27",
        "pydantic>=2.6,<3",
    )
    .add_local_dir(str(REPO_ROOT / "ui"), "/app/ui")
    .add_local_dir(str(REPO_ROOT / "data"), "/app/data")
    .add_local_dir(str(REPO_ROOT / "analysis"), "/app/analysis")
    .add_local_dir(str(REPO_ROOT / "rag"), "/app/rag")
    .add_local_dir(str(REPO_ROOT / "api"), "/app/api")
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
)

corpus_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_common_pip)
    .add_local_dir(str(REPO_ROOT), "/app", ignore=["**/.venv", "**/__pycache__", "**/.git"])
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
)


# --------------------------------------------------------------------------
# Shared secrets + volumes
# --------------------------------------------------------------------------

# `accord` secret carries DATABASE_URL (Neon), LANGFUSE_* keys, and (optionally)
# SGLANG_MODEL to override the default model checkpoint.
accord_secret = modal.Secret.from_name("accord")

# HF model cache — one Volume shared by all GPU containers so the 15 GB
# Qwen2.5-7B FP16 download happens once, not on every cold start.
hf_cache = modal.Volume.from_name("accord-hf-cache", create_if_missing=True)

# Trained artifacts (outcome model, calibrator). Written by `train_outcome`,
# read by the GPU container on-demand.
artifacts_volume = modal.Volume.from_name("accord-artifacts", create_if_missing=True)


# --------------------------------------------------------------------------
# Main GPU app: SGLang + FastAPI colocated
# --------------------------------------------------------------------------

app = modal.App("accord")


@app.cls(
    image=gpu_image,
    gpu="H100",
    secrets=[accord_secret],
    volumes={
        HF_CACHE_DIR: hf_cache,
        # analysis/outcome_service.py reads models/outcome_model.joblib by
        # default; keep the Volume mount aligned so the same code works
        # locally and on Modal.
        "/app/models": artifacts_volume,
    },
    max_containers=1,
    scaledown_window=300,
    timeout=1800,  # ASGI request timeout — recommendation call can take ~10s cold
)
@modal.concurrent(max_inputs=4)
class AccordServer:
    """SGLang server + FastAPI ASGI, one container per Modal input burst."""

    @modal.enter()
    def start_sglang(self) -> None:
        """Boot SGLang on 127.0.0.1:30000 and block until the /v1/models endpoint answers."""
        import socket

        model = os.environ.get("SGLANG_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        # SGLang launcher — --host 127.0.0.1 keeps the port container-local.
        self._sglang = subprocess.Popen(
            [
                "python", "-m", "sglang.launch_server",
                "--model-path", model,
                "--host", "127.0.0.1",
                "--port", "30000",
                # Structured-output guarantees for LangChain's .with_structured_output.
                "--grammar-backend", "xgrammar",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # Wait for the port to accept connections (~60-90 s from a cached weights volume).
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", 30000), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("SGLang failed to open port 30000 within 900 s")

        # Confirm the HTTP layer is up before ASGI accepts traffic.
        import httpx

        for _ in range(60):
            try:
                r = httpx.get("http://127.0.0.1:30000/v1/models", timeout=2.0)
                if r.status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

        os.environ.setdefault("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1")

    @modal.exit()
    def stop_sglang(self) -> None:
        proc = getattr(self, "_sglang", None)
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                proc.kill()

    @modal.asgi_app()
    def api(self):
        """Return the FastAPI app — Modal serves it at https://<workspace>--accord-accordserver-api.modal.run."""
        from api.main import build_app

        return build_app()


# --------------------------------------------------------------------------
# One-shot: embed the case corpus into Neon
# --------------------------------------------------------------------------


@app.function(
    image=corpus_image,
    secrets=[accord_secret],
    timeout=1800,
)
def build_corpus() -> int:
    """Run `rag.embed.embed_case_corpus` inside Modal against Neon.

    Assumes `data/processed/case_corpus.jsonl` was built locally (`python -m
    data.build_case_corpus`) and shipped in the image via `add_local_dir`.
    Returns the number of documents upserted.
    """
    from rag.embed import embed_case_corpus

    n = embed_case_corpus()
    print(f"[accord] upserted {n} documents into Neon pgvector.")
    return n


# --------------------------------------------------------------------------
# One-shot: train the XGBoost outcome model, save to artifacts Volume
# --------------------------------------------------------------------------


@app.function(
    image=corpus_image,
    volumes={"/app/models": artifacts_volume},
    timeout=1800,
)
def train_outcome() -> str:
    """Train + calibrate + save the outcome model to the artifacts Volume.

    Writes to `/app/models/outcome_model.joblib` — the path
    `analysis/outcome_service.py` reads by default, so the GPU container
    (which mounts the same Volume at `/app/models`) picks it up on next request.
    """
    import json

    from analysis.outcome_model import (
        build_feature_matrix,
        calibrate,
        save_model,
        train_outcome_model,
    )
    from data.schema import Transcript

    processed = Path("/app/data/processed/craigslist_bargain.jsonl")
    if not processed.exists():
        raise RuntimeError(
            f"{processed} missing — run `python -m data.ingest_craigslist --download` locally, "
            "then re-deploy so add_local_dir picks the file up."
        )

    # Load transcripts and split by the source-provided split.
    train, val = [], []
    with processed.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = Transcript.model_validate_json(line)
            split = t.metadata.get("split", "train")
            (train if split == "train" else val if split == "validation" else train).append(t)

    X_train, y_train = build_feature_matrix(train)
    X_val, y_val = build_feature_matrix(val or train[: max(1, len(train) // 5)])

    model = train_outcome_model(X_train, y_train)
    calibrator = calibrate(model, X_val, y_val)

    out = Path("/app/models/outcome_model.joblib")
    save_model(model, calibrator, list(X_train.columns), out)
    artifacts_volume.commit()

    meta = {"train_rows": len(X_train), "val_rows": len(X_val), "path": str(out)}
    print(json.dumps(meta))
    return str(out)


# --------------------------------------------------------------------------
# UI app: Streamlit
# --------------------------------------------------------------------------

ui_app = modal.App("accord-ui")


@ui_app.function(
    image=cpu_image,
    # `accord` secret must include `ACCORD_API_URL` — the deployed FastAPI URL.
    # Add it after the first `modal deploy` of the `accord` app:
    #   modal secret update accord ACCORD_API_URL=https://<workspace>--accord-accordserver-api.modal.run
    secrets=[accord_secret],
    max_containers=1,
    scaledown_window=300,
    timeout=600,
)
@modal.web_server(port=8501, startup_timeout=60)
def ui() -> None:
    """Start Streamlit; Modal proxies HTTPS → localhost:8501 as long as traffic keeps flowing."""
    subprocess.Popen(
        [
            "streamlit", "run", "/app/ui/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
    )
