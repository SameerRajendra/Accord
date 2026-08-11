# Accord — Cloud deploy sequence

This file is the concrete sequence for standing Accord up on Modal + Neon +
Langfuse and producing the two URLs you can share with a reviewer. See
[DESIGN.md](DESIGN.md) for the architecture that produced these steps.

**Prerequisites:** a Modal account (`pip install modal && modal token new`),
a Neon account, a Langfuse cloud account (free tier is enough).

---

## 0. Local prep (once)

```bash
# From the repo root:
python -m data.ingest_casino --download   # -> data/processed/casino.jsonl
python -m data.build_case_corpus               # -> data/processed/case_corpus.jsonl
```

These files ship into Modal via `add_local_dir` (see
[infra/modal/app.py](infra/modal/app.py)), so they must exist before deploy.

---

## 1. Neon: create DB, install pgvector

Follow [infra/neon/README.md](infra/neon/README.md). At the end you have a
`DATABASE_URL` string and the `vector` extension installed.

---

## 2. Langfuse: get keys

1. https://cloud.langfuse.com → new project → Settings → API keys.
2. Copy the **Public** and **Secret** keys.
3. Note the host: `https://cloud.langfuse.com`.

Callback wiring is env-gated — omit these keys entirely to run without
tracing (`agent/callbacks.py`).

---

## 3. Modal: wire the secret

```bash
modal secret create accord \
  DATABASE_URL="postgresql://user:pw@host-pooler.region.aws.neon.tech/neondb?sslmode=require" \
  LANGFUSE_PUBLIC_KEY="pk-lf-..." \
  LANGFUSE_SECRET_KEY="sk-lf-..." \
  LANGFUSE_HOST="https://cloud.langfuse.com"
```

`ACCORD_API_URL` will be added to this same secret after step 5.

---

## 4. Populate Neon + train the outcome model

Both are one-shot Modal functions that use the `accord` secret and the shared
Volumes declared in [infra/modal/app.py](infra/modal/app.py):

```bash
modal run infra/modal/app.py::build_corpus       # embeds case corpus → Neon
modal run infra/modal/app.py::train_outcome      # trains XGBoost → accord-artifacts Volume
```

After `build_corpus` finishes, re-run `rag/schema.sql` so the HNSW index
gets created on the now-populated embedding table:

```bash
psql "$DATABASE_URL" -f rag/schema.sql
```

Verify:

```bash
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM langchain_pg_embedding;"
```

---

## 5. Deploy

```bash
modal deploy infra/modal/app.py
```

This deploys the single `accord` app with all four entrypoints (GPU class +
FastAPI ASGI, Streamlit `ui`, and the two one-shot functions). Two URLs are
printed — the API:

```
https://<workspace>--accord-accordserver-api.modal.run
```

…and the UI (`https://<workspace>--accord-ui.modal.run`).

Update the secret so the UI knows how to reach the API:

```bash
modal secret update accord ACCORD_API_URL="https://<workspace>--accord-accordserver-api.modal.run"
modal deploy infra/modal/app.py                  # redeploy so the UI picks up the new secret
```

---

## 6. Smoke-test

```bash
# Health check (returns immediately, does NOT wait for SGLang if cold).
curl https://<workspace>--accord-accordserver-api.modal.run/health

# Full analysis — first request pays ~60–90 s cold start; second request is warm.
curl -X POST https://<workspace>--accord-accordserver-api.modal.run/analyze \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

Or open the UI in a browser and paste a transcript:

```
https://<workspace>--accord-ui.modal.run
```

That's the URL to share with Yash.

---

## 7. Observability

Langfuse dashboard (https://cloud.langfuse.com) shows one trace per
`/analyze` call, with each LangGraph node (`sentiment`, `behaviors`,
`outcome`, `retrieve`, `recommend`) as a nested span. Useful for reviewing
the actual prompts + structured-output responses without spelunking logs.

---

## 8. Cost sanity

- **Idle:** ~$0. Modal scales to zero after `scaledown_window=300s`.
- **Per warm request:** ~2–8 GPU-seconds on H100 ≈ ~$0.003–0.012 (Modal H100
  ≈ $0.001–0.002/s at time of writing — check the Modal pricing page).
- **Cold start:** ~60–90 GPU-seconds ≈ ~$0.10–0.20 per cold hit.
- **Neon:** free tier.
- **Langfuse:** free tier.

Two demo sessions a day (~20 requests each, most warm) is well under $2/mo.

---

## 9. Tear down

```bash
modal app stop accord
# (one app now — `accord` covers the API, the UI, and both one-shot functions)
# Volumes and Secrets persist — delete via `modal volume delete` /
# `modal secret delete` if you want a full wipe.
```
