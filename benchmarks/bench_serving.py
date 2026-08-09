"""Serving benchmark for Accord's SGLang runtime.

Produces the three numbers DESIGN.md §7 promises and `results/` does not yet have:

  1. cold-start vs warm      -> results/coldstart.csv
  2. throughput/concurrency  -> results/batching_curve.csv
  3. p50/p95 + $/request     -> results/latency_cost.csv

Runs INSIDE the GPU container and drives SGLang over localhost. This is
deliberate: the deployed ASGI path is capped at `max_inputs=4` /
`max_containers=1`, so load driven from outside the container measures Modal's
admission control, not SGLang's batching. Hitting localhost:30000 directly
measures the serving engine, which is what the batching curve is about.

Usage
-----
    # pass 1 — debug the harness cheap (~$1)
    modal run benchmarks/bench_serving.py::bench --gpu-type A10G \
        --model Qwen/Qwen2.5-1.5B-Instruct --concurrencies 1,2,4,8

    # pass 2 — the real run (~$5)
    modal run benchmarks/bench_serving.py::bench --gpu-type H100 \
        --model Qwen/Qwen2.5-7B-Instruct --concurrencies 1,2,4,8,16,32,64

CSVs land in the `accord-artifacts` Volume and are echoed to stdout; copy them
into `results/` and commit.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Image — mirrors infra/modal/app.py's gpu_image, minus the app-layer deps.
# Kept self-contained so this file can't break the deploy.
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Official SGLang runtime image, immutable tag. Building the stack ourselves
# (pip install sglang[all]==0.4.1 + unpinned transformers) breaks: modern
# transformers no longer exposes AutoProcessor at the top level, and pip pulls
# torch cu121 rather than the cu124 build flashinfer's wheels expect, silently
# disabling the compiled kernels. The upstream image resolves all of that.
# `-runtime` excludes build/dev tooling (~40% smaller than the full image).
SGLANG_IMAGE = "lmsysorg/sglang:v0.5.16-cu129-runtime"

# The SGLang image already has content under /root/.cache/huggingface, and Modal
# refuses to mount a Volume over a non-empty path. Relocate the HF cache instead
# of clearing theirs — HF_HOME keeps the volume's internal layout identical.
HF_CACHE_DIR = "/cache/huggingface"

bench_image = (
    modal.Image.from_registry(SGLANG_IMAGE)
    .entrypoint([])  # image ships its own ENTRYPOINT; Modal needs it cleared
    .pip_install("httpx>=0.27")
    .env({"HF_HOME": HF_CACHE_DIR})
    .workdir("/app")
)

app = modal.App("accord-bench")

hf_cache = modal.Volume.from_name("accord-hf-cache", create_if_missing=True)
artifacts_volume = modal.Volume.from_name("accord-artifacts", create_if_missing=True)

# Modal on-demand rates, $/hr. VERIFY against https://modal.com/pricing before
# citing any $/request number in the README — these move.
GPU_HOURLY = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "L40S": 1.95,
    "H100": 4.56,
}

PORT = 30000
BASE = f"http://127.0.0.1:{PORT}/v1"


# --------------------------------------------------------------------------
# SGLang lifecycle
# --------------------------------------------------------------------------


SGLANG_LOG = Path("/tmp/sglang.log")


def _tail(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log captured)"
    return "\n".join(lines[-n:])


def _launch_sglang(model: str, extra_args: str = "") -> tuple[subprocess.Popen, float]:
    """Start SGLang; return (process, seconds until /v1/models answered 200).

    Launcher args mirror infra/modal/app.py so the measurement reflects the
    deployed configuration. Output is captured to SGLANG_LOG and echoed on
    failure — a silent crash is not debuggable on a metered GPU.
    """
    import httpx

    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ]
    if extra_args.strip():
        cmd += extra_args.split()

    print(f"[bench] launching: {' '.join(cmd)}")

    t0 = time.perf_counter()
    log_fh = SGLANG_LOG.open("w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)

    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            log_fh.flush()
            raise RuntimeError(
                f"SGLang exited early with code {proc.returncode}.\n"
                f"--- last {80} lines of SGLang output ---\n{_tail(SGLANG_LOG)}"
            )
        try:
            r = httpx.get(f"{BASE}/models", timeout=2.0)
            if r.status_code == 200:
                return proc, time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)

    log_fh.flush()
    raise RuntimeError(
        f"SGLang did not become ready within 900 s.\n"
        f"--- last {80} lines of SGLang output ---\n{_tail(SGLANG_LOG)}"
    )


def _detect_gpu() -> str:
    """Read the actual GPU from nvidia-smi and map it to a GPU_HOURLY key.

    The `gpu_type` CLI arg only labels output — Modal provisions from the
    decorator, which is resolved at import time. Trusting the arg silently
    produced A10G pricing on H100 hardware, so the CSV records what actually ran.
    """
    try:
        name = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"

    upper = name.upper()
    # Order matters: A100 before A10 ("A100" contains "A10"), L40S before L4.
    if "H200" in upper:
        return "H200"
    if "H100" in upper:
        return "H100"
    if "L40S" in upper:
        return "L40S"
    if "A100" in upper:
        return "A100-80GB" if "80" in upper else "A100-40GB"
    if "A10" in upper:  # Modal's A10G reports as "NVIDIA A10"
        return "A10G"
    if "L4" in upper:
        return "L4"
    if "T4" in upper:
        return "T4"
    return name


def _make_prompt(approx_tokens: int) -> str:
    """Synthetic prompt of roughly `approx_tokens` tokens (~0.75 tok/word)."""
    unit = (
        "The buyer opened well below asking and the seller held firm on price "
        "while signalling flexibility on the pickup date. "
    )
    words_needed = int(approx_tokens / 0.75)
    reps = max(1, words_needed // len(unit.split()) + 1)
    return (unit * reps).strip()


# --------------------------------------------------------------------------
# Request driver
# --------------------------------------------------------------------------


async def _one_request(
    client, model: str, prompt: str, max_tokens: int, ignore_eos: bool = True
) -> dict:
    """Stream one completion; return TTFT, e2e latency, and output token count.

    Output tokens are counted as streamed content chunks. SGLang emits roughly
    one token per chunk, so this is an approximation — stated rather than hidden.

    `ignore_eos` forces every request to generate exactly `max_tokens`. Without
    it the model stops at EOS and output length varies with batch composition,
    which confounds tok/s across concurrency levels — the whole point of the curve.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    start = time.perf_counter()
    ttft = None
    n_out = 0

    async with client.stream("POST", f"{BASE}/chat/completions", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line[6:].strip()
            if body == "[DONE]":
                break
            try:
                delta = json.loads(body)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - start
                n_out += 1

    e2e = time.perf_counter() - start
    return {"ttft": ttft if ttft is not None else e2e, "e2e": e2e, "out_tokens": n_out}


async def _run_level(
    model: str,
    concurrency: int,
    n_requests: int,
    in_tokens: int,
    out_tokens: int,
    ignore_eos: bool = True,
    unique_prompts: bool = True,
) -> tuple[dict, list[dict]]:
    """Drive `n_requests` at fixed `concurrency`; return (aggregate, per-request)."""
    import httpx

    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 8)

    async with httpx.AsyncClient(timeout=600.0, limits=limits) as client:
        wall_start = time.perf_counter()

        async def guarded(i: int):
            async with sem:
                started = time.perf_counter() - wall_start
                # Unique prefix per request. With one shared prompt, SGLang's
                # RadixAttention serves every request from the prefix cache and
                # the curve measures the cached path, not realistic traffic.
                p = f"[req {i:05d}] " + _make_prompt(in_tokens) if unique_prompts else _make_prompt(in_tokens)
                r = await _one_request(client, model, p, out_tokens, ignore_eos)
                r["index"] = i
                r["start_offset_s"] = round(started, 4)
                return r

        results = await asyncio.gather(*[guarded(i) for i in range(n_requests)])
        wall = time.perf_counter() - wall_start

    ttfts = sorted(r["ttft"] for r in results)
    e2es = sorted(r["e2e"] for r in results)
    total_out = sum(r["out_tokens"] for r in results)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, int(round(p * (len(xs) - 1))))
        return xs[idx]

    agg = {
        "concurrency": concurrency,
        "requests": n_requests,
        "wall_s": round(wall, 3),
        "ttft_p50_ms": round(pct(ttfts, 0.50) * 1000, 1),
        "ttft_p95_ms": round(pct(ttfts, 0.95) * 1000, 1),
        "e2e_p50_ms": round(pct(e2es, 0.50) * 1000, 1),
        "e2e_p95_ms": round(pct(e2es, 0.95) * 1000, 1),
        "e2e_mean_ms": round(statistics.fmean(e2es) * 1000, 1),
        "output_tok_s": round(total_out / wall, 1),
        "requests_s": round(n_requests / wall, 3),
        "total_output_tokens": total_out,
    }

    per_request = [
        {
            "concurrency": concurrency,
            "index": r["index"],
            "start_offset_s": r["start_offset_s"],
            "ttft_ms": round(r["ttft"] * 1000, 1),
            "e2e_ms": round(r["e2e"] * 1000, 1),
            "out_tokens": r["out_tokens"],
        }
        for r in sorted(results, key=lambda x: x["index"])
    ]
    return agg, per_request


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------


def _emit_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


@app.function(
    image=bench_image,
    gpu=os.environ.get("BENCH_GPU", "H100"),
    volumes={HF_CACHE_DIR: hf_cache, "/app/results": artifacts_volume},
    timeout=3600,
    # Short scaledown: idle H100 after a benchmark is pure waste.
    scaledown_window=60,
)
def bench(
    gpu_type: str = "H100",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    concurrencies: str = "1,2,4,8,16,32,64",
    requests_per_level: int = 128,
    in_tokens: int = 512,
    out_tokens: int = 128,
    warmup: int = 8,
    ignore_eos: bool = True,
    unique_prompts: bool = True,
    sglang_args: str = "--grammar-backend xgrammar",
) -> str:
    """Cold start, batching curve, and $/request in one container lifetime."""
    levels = [int(c) for c in concurrencies.split(",") if c.strip()]

    subprocess.run(["nvidia-smi"], check=False)

    # Provisioning comes from the decorator (BENCH_GPU), not from --gpu-type.
    # Always label and price from the hardware actually present.
    detected = _detect_gpu()
    if detected != gpu_type:
        print(
            f"[bench] WARNING: --gpu-type={gpu_type} but nvidia-smi reports {detected}. "
            f"Using {detected} for labels and cost. Set BENCH_GPU={gpu_type} before "
            f"`modal run` to change what Modal actually provisions."
        )
    gpu_type = detected
    hourly = GPU_HOURLY.get(gpu_type)

    # --- 1. cold start -----------------------------------------------------
    proc, cold_s = _launch_sglang(model, sglang_args)
    print(f"[bench] SGLang ready in {cold_s:.1f} s on {gpu_type} ({model})")

    try:
        # --- warmup (excluded from all reported numbers) -------------------
        asyncio.run(
            _run_level(model, min(2, levels[0]), warmup, in_tokens, out_tokens, ignore_eos,
                       unique_prompts)
        )

        # --- 2. batching curve --------------------------------------------
        rows = []
        traces: list[dict] = []
        for c in levels:
            # Per-level warmup. Per-request tracing showed the p95 outlier at a
            # given concurrency was always the first wave — SGLang pays a
            # one-time cost the first time it sees a batch size (CUDA graph
            # capture / KV pool allocation). Warming each level separately keeps
            # that transient out of the reported steady-state distribution.
            asyncio.run(
                _run_level(model, c, c * 2, in_tokens, out_tokens, ignore_eos, unique_prompts)
            )

            # ≥8 waves per level so p95 is over a real distribution, not 2 bursts.
            n = max(requests_per_level, c * 8)
            row, trace = asyncio.run(
                _run_level(model, c, n, in_tokens, out_tokens, ignore_eos, unique_prompts)
            )
            traces.extend(trace)
            row["gpu"] = gpu_type
            row["model"] = model
            row["in_tokens"] = in_tokens
            row["out_tokens_requested"] = out_tokens
            row["ignore_eos"] = ignore_eos

            # --- 3. cost --------------------------------------------------
            if hourly:
                gpu_seconds = row["wall_s"]
                row["usd_per_1k_requests"] = round(
                    (gpu_seconds * hourly / 3600) / row["requests"] * 1000, 4
                )
                row["usd_per_1m_output_tokens"] = round(
                    (gpu_seconds * hourly / 3600)
                    / max(1, row["total_output_tokens"])
                    * 1_000_000,
                    2,
                )
            rows.append(row)
            print(f"[bench] c={c:<3} {json.dumps(row)}")

        # --- write artifacts ----------------------------------------------
        out_dir = Path("/app/results")
        out_dir.mkdir(parents=True, exist_ok=True)

        curve_csv = _emit_csv(rows)
        (out_dir / "batching_curve.csv").write_text(curve_csv)

        # Per-request trace — needed to tell a warmup transient apart from a
        # periodic stall when an aggregate p95 looks wrong.
        if traces:
            (out_dir / "per_request_trace.csv").write_text(_emit_csv(traces))
            slow = sorted(traces, key=lambda r: -r["ttft_ms"])[:10]
            print("\n[bench] 10 slowest requests by TTFT:")
            for r in slow:
                print(f"  c={r['concurrency']:<3} idx={r['index']:<5} "
                      f"start={r['start_offset_s']:>7.2f}s  ttft={r['ttft_ms']:>8.1f}ms  "
                      f"e2e={r['e2e_ms']:>8.1f}ms")

        warm_p50 = rows[0]["e2e_p50_ms"] / 1000
        cold_rows = [{
            "gpu": gpu_type,
            "model": model,
            "cold_start_s": round(cold_s, 1),
            "first_warm_request_p50_s": round(warm_p50, 3),
            "cold_start_usd": round(cold_s * hourly / 3600, 4) if hourly else "",
            "note": "cold = container enter -> /v1/models 200; excludes image pull",
        }]
        (out_dir / "coldstart.csv").write_text(_emit_csv(cold_rows))

        artifacts_volume.commit()

        print("\n===== coldstart.csv =====\n" + _emit_csv(cold_rows))
        print("===== batching_curve.csv =====\n" + curve_csv)
        return curve_csv

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            proc.kill()
