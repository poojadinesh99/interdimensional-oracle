# interdimensional-oracle

A lightweight scaffold for an oracle-style RAG app with a Python backend and static frontend.

## Structure

- `backend/app/`: Flask app, config, retrieval, prompts, and guardrails
- `backend/app/data/`: generated data artifacts and ingestion helper
- `backend/tests/`: basic backend tests
- `frontend/`: static UI

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The oracle can run in three modes, set via `ORACLE_MODE` in `.env`:

- `local` (default) — answers straight from the wisdom corpus via retrieval.
  No API calls, no cost, no setup beyond the index build below.
- `ollama` — generates answers with a local model through
  [Ollama](https://ollama.com), also free. Install Ollama, run
  `ollama serve`, then `ollama pull llama3.2:3b` (or set `OLLAMA_MODEL` to
  whichever model you've pulled).
- `anthropic` — calls the Claude API for generated answers. Needs a funded
  `ANTHROPIC_API_KEY` from the [Anthropic Console](https://console.anthropic.com/).

Then build the retrieval index (bag-of-words vectors over the wisdom corpus
in `backend/app/data/corpus.json`) — required in all modes:

```bash
python -m backend.app.data.ingest
```

## Run

```bash
python -m backend.app.main
```

Then open `http://127.0.0.1:5050`. (On macOS, port 5000 is often taken by
the AirPlay Receiver, so this project defaults to 5050 — override with
`PORT` in `.env` if you want a different one.)

## Test

```bash
pytest
```