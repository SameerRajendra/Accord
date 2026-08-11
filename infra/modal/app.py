"""Modal deploy for Accord: SGLang + FastAPI + Streamlit, scale-to-zero.

**One** Modal app (`accord`) with four entrypoints. It has to be one app, not
two: `modal deploy <file>` resolves a single `modal.App` per file, so a second
`App` object here would make the documented one-command deploy fail.

1. **`AccordServer`** — GPU class (`H100`, `max_containers=1`,
   `scaledown_window=300s`). Starts SGLang serving Qwen2.5-7B-Instruct on
   container-enter, exposes the FastAPI `/analyze` + `/health` endpoints as a
   Modal ASGI app on the *same* container so inference calls stay on localhost.
   Weights are cached on a Modal Volume so cold starts don't re-download 15 GB.
2. **`ui`** — CPU-only Streamlit web server. Calls the API over HTTPS via the
   `ACCORD_API_URL` secret. Separate image (`cpu_image`), so UI edits don't
   rebuild the GPU image layers.
3. **`build_corpus`** — one-shot; embeds the case corpus into Neon.
4. **`train_outcome`** — one-shot; trains + saves the XGBoost artifact onto the
   artifacts Volume that `AccordServer` mounts at `/app/models`.

Deploy sequence — see [RUN.md](../../RUN.md) for the full sequence including
Neon and Langfuse setup.

    modal secret create accord DATABASE_URL=postgresql://... \\
                                LANGFUSE_PUBLIC_KEY=... \\
                                LANGFUSE_SECRET_KEY=... \\
                                LANGFUSE_HOST=https://cloud.langfuse.com

    modal run infra/modal/app.py::build_corpus       # one-shot, populates Neon
    modal run infra/modal/app.py::train_outcome      # one-shot, saves outcome_model.joblib
    modal deploy infra/modal/app.py                  # deploys all four entrypoints
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

def _repo_root() -> Path:
    """Repo root locally; a harmless stand-in inside a Modal container.

    Modal copies this file to `/root/app.py` in the container, so
    `Path(__file__).resolve().parents[2]` raises IndexError there — `/root`
    has only two ancestors. The image definitions below are only *used* at
    build time (which always happens locally), but they're module-level
    statements, so they still execute when Modal imports this module inside
    the container to find the function being invoked.
    """
    here = Path(__file__).resolve()
    if len(here.parents) >= 3:
        return here.parents[2]
    return Path("/app")


REPO_ROOT = _repo_root()

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
        # The base image ships some distro-packaged Python deps (PyJWT) with no
        # dist-info RECORD, so pip cannot uninstall them when a dependency wants
        # a different version — it aborts with "uninstall-no-record-file".
        # Reinstalling under pip's control first gives them a RECORD, after
        # which the real install can upgrade them normally.
        "pip install --ignore-installed --no-deps PyJWT",
    )
    .pip_install(*_common_pip, extra_options="-c /tmp/sglang-constraints.txt")
    .workdir("/app")
    .env({"PYTHONPATH": "/app", "HF_HOME": HF_CACHE_DIR})
    # add_local_* MUST come last. Modal forbids any build step after it — and
    # .workdir()/.env() count as build steps. Keeping it last also means local
    # edits don't invalidate the expensive pip layers above.
    #
    # `models/` and `results/` are excluded: the artifacts Volume mounts at
    # /app/models at runtime, and Modal refuses to mount a Volume over a
    # non-empty path. `models/outcome_model.joblib` exists locally after an
    # eval run, so without this exclusion the mount fails and the deploy dies.
    .add_local_dir(
        str(REPO_ROOT),
        "/app",
        ignore=[
            "**/.venv", "**/__pycache__", "**/.git", "**/node_modules",
            "models", "models/**",
            "results", "results/**",
            "data/raw", "data/raw/**",
        ],
    )
)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "streamlit>=1.36",
        "httpx>=0.27",
        "pydantic>=2.6,<3",
    )
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    # add_local_* last — see the gpu_image comment. The UI only needs its own
    # module; it talks to the API over HTTPS, not by importing the pipeline.
    .add_local_dir(str(REPO_ROOT / "ui"), "/app/ui")
)

corpus_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Same distro-packaged-PyJWT problem as gpu_image — see that comment.
    .run_commands("pip install --ignore-installed --no-deps PyJWT")
    .pip_install(*_common_pip)
    .workdir("/app")
    .env({"PYTHONPATH": "/app"})
    # add_local_* last (Modal forbids build steps after it). Same
    # Volume-over-non-empty-path constraint as gpu_image: `train_outcome`
    # mounts the artifacts Volume at /app/models, so /app/models must not be
    # baked into the image.
    .add_local_dir(
        str(REPO_ROOT),
        "/app",
        ignore=[
            "**/.venv", "**/__pycache__", "**/.git",
            "models", "models/**",
            "results", "results/**",
            "data/raw", "data/raw/**",
        ],
    )
)


# --------------------------------------------------------------------------
# Shared secrets + volumes
# --------------------------------------------------------------------------

# `accord` secret carries DATABASE_URL (Neon), LANGFUSE_* keys, and (optionally)
# SGLANG_MODEL to override the default model checkpoint.
accord_secret = modal.Secret.from_name("accord")

# The UI's API URL lives in its OWN secret, deliberately.
#
# Modal has no `secret update` — you re-create with `--force`, which replaces
# the WHOLE secret. When ACCORD_API_URL shared the `accord` secret, adding it
# after the first deploy silently wiped DATABASE_URL, and /analyze started
# failing with a DNS error on a placeholder host. Separating them means
# updating the UI URL can never clobber the database credentials.
#
# It's also least-privilege: the UI container talks to the API over HTTPS and
# never touches Postgres, so it has no reason to hold the Neon password.
accord_ui_secret = modal.Secret.from_name("accord-ui")

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
        # stdout/stderr are inherited so SGLang's logs land in Modal's log
        # stream. Swallowing them (DEVNULL) makes a failed cold start
        # undiagnosable — the container just times out with no explanation.
        self._sglang = subprocess.Popen(
            [
                "python", "-m", "sglang.launch_server",
                "--model-path", model,
                "--host", "127.0.0.1",
                "--port", "30000",
                # Structured-output guarantees for LangChain's .with_structured_output.
                "--grammar-backend", "xgrammar",
            ],
        )

        def _assert_alive(stage: str) -> None:
            """Fail fast if SGLang died, instead of waiting out the full timeout."""
            code = self._sglang.poll()
            if code is not None:
                raise RuntimeError(f"SGLang exited during {stage} with code {code}")

        # Phase 1: wait for the port to bind. This only proves the process
        # started — weights are typically still loading.
        deadline = time.time() + 900
        while time.time() < deadline:
            _assert_alive("startup")
            try:
                with socket.create_connection(("127.0.0.1", 30000), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("SGLang failed to open port 30000 within 900 s")

        # Phase 2: the real readiness gate — /v1/models returns 200 only once
        # the model is actually loaded. The `else: raise` matters: without it a
        # backend that never comes up falls through silently and the ASGI app
        # starts serving traffic against a dead SGLang, turning a clean startup
        # failure into confusing per-request 500s.
        import httpx

        for _ in range(90):
            _assert_alive("model load")
            try:
                if httpx.get("http://127.0.0.1:30000/v1/models", timeout=2.0).status_code == 200:
                    break
            except Exception:  # noqa: BLE001 — not accepting HTTP yet
                pass
            time.sleep(2)
        else:
            raise RuntimeError(
                "SGLang bound port 30000 but /v1/models never returned 200 within 180 s"
            )

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

    processed = Path("/app/data/processed/casino.jsonl")
    if not processed.exists():
        raise RuntimeError(
            f"{processed} missing — run `python -m data.ingest_casino --download` locally, "
            "then re-deploy so add_local_dir picks the file up."
        )

    # Load transcripts, keeping the splits STRICTLY separate. An earlier version
    # of this fell back to `train` for anything that wasn't "validation", which
    # silently appended the *test* split into the training set — the deployed
    # model was training on its own held-out data. Rows with a missing or
    # unrecognized split are dropped rather than defaulted into training.
    by_split = {"train": [], "validation": [], "test": []}
    skipped = 0
    with processed.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = Transcript.model_validate_json(line)
            split = t.metadata.get("split")
            if split in by_split:
                by_split[split].append(t)
            else:
                skipped += 1

    train, val = by_split["train"], by_split["validation"]
    if not train:
        raise RuntimeError(
            f"no rows with split='train' in {processed} "
            f"({skipped} rows had a missing/unknown split)"
        )
    if not val:
        raise RuntimeError(
            "no rows with split='validation' — isotonic calibration needs a held-out "
            "split, and calibrating on training data would produce a meaningless curve"
        )

    X_train, y_train = build_feature_matrix(train)
    X_val, y_val = build_feature_matrix(val)

    model = train_outcome_model(X_train, y_train)
    calibrator = calibrate(model, X_val, y_val)

    out = Path("/app/models/outcome_model.joblib")
    save_model(model, calibrator, list(X_train.columns), out)
    artifacts_volume.commit()

    meta = {"train_rows": len(X_train), "val_rows": len(X_val), "path": str(out)}
    print(json.dumps(meta))
    return str(out)


# --------------------------------------------------------------------------
# Streamlit UI — same app, separate (CPU) image
# --------------------------------------------------------------------------

@app.function(
    image=cpu_image,
    # Only the UI URL — no database credentials. See accord_ui_secret above.
    #   modal secret create accord-ui ACCORD_API_URL=https://<workspace>--accord-accordserver-api.modal.run
    secrets=[accord_ui_secret],
    max_containers=1,
    scaledown_window=300,
    # Long-lived: Streamlit's /_stcore/stream websocket is a single request that
    # stays open for the whole session, and it is killed when this timeout
    # expires. 600 s would drop a user's UI after 10 minutes.
    timeout=3600,
)
# Streamlit's frontend pulls ~60 separate JS chunks before it can hydrate
# widgets. Without this, the container serves them near-serially: logs showed
# `execution: ~124 ms` but `duration: 8-17 s` per chunk (pure queueing), and a
# `GET / -> 200 OK (duration: 119.5 s)`. The page rendered its text immediately
# and left every widget as an empty skeleton for minutes while chunks trickled
# in. High concurrency is safe here — serving static assets is IO-bound, not
# CPU-bound, and this container does no inference.
@modal.concurrent(max_inputs=100)
@modal.web_server(port=8501, startup_timeout=120)
def ui() -> None:
    """Start Streamlit; Modal proxies HTTPS → localhost:8501 as long as traffic keeps flowing."""
    subprocess.Popen(
        [
            "streamlit", "run", "/app/ui/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            # NOTE: --server.enableCORS=false / --server.enableXsrfProtection=false
            # are the usual "Streamlit behind a proxy" advice and were tried here
            # first. They are NOT needed: Modal's proxy upgrades the websocket
            # cleanly (`CONNECT /_stcore/stream -> 101 Switching Protocols` in the
            # logs). The empty-widget symptom was request queueing, fixed by the
            # @modal.concurrent above — so these stay ON rather than weakening
            # XSRF protection for a problem that never existed.
        ]
    )
